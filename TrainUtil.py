from abc import ABC, abstractmethod
import os.path
import random
import re
import sys
import argparse

import fairseq.models.transformer
import torch
import textdistance
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import wandb
import numpy as np

from sklearn.model_selection import train_test_split

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BartTokenizer, BartForConditionalGeneration
from torch.optim import AdamW

from operator import itemgetter

from DataUtil import load_data
import Common


# This code is from https://colab.research.google.com/github/jaygala24/pytorch-implementations/blob/master/Attention%20Is%20All%20You%20Need.ipynb#scrollTo=TWOqbtFNzcU_
class NoamOptim(object):
    """ Optimizer wrapper for learning rate scheduling.
    """

    def __init__(self, optimizer, d_model, factor=2, n_warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.factor = factor
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.n_steps += 1
        lr = self.get_lr()
        for p in self.optimizer.param_groups:
            p['lr'] = lr
        self.optimizer.step()

    def get_lr(self):
        return self.factor * (
                self.d_model ** (-0.5)
                * min(self.n_steps ** (-0.5), self.n_steps * self.n_warmup_steps ** (-1.5))
        )

def get_attended_token_count(mask):
    # expecting shape (batch_size, seq_length)
    return torch.count_nonzero(mask, dim=1)


def sort_by_length(a, masks, c, d):
    sorted_tuples = sorted(zip(a, masks, c, d), key=lambda x: torch.count_nonzero(
        x[1]))  # sort by number of non-zero entries in the input-masks
    sorted_tensors = (torch.stack(list(map(lambda t: t[i], sorted_tuples))) for i in range(len(sorted_tuples[0])))
    return sorted_tensors

class Metrics(ABC):
    @abstractmethod
    def update(self, batch, outputs):
        pass

    @abstractmethod
    def print(self, index, set_size, prefix, count, epoch, loss, decoder=None):
        pass

    @abstractmethod
    def reset(self):
        pass


def get_bracket_depth(bracketed_string: str):
    max_count = 0
    current_open = 0
    for c in bracketed_string:
        if c == "(":
            current_open += 1
            max_count = max(max_count, current_open)
        elif c == ")":
            current_open -= 1
    return max_count



class Trainer(ABC):
    def __init__(self, hp):
        self.hp = hp

    def decode_tokens(self, tokenizer, ids) -> list[str]:
        raise NotImplementedError("This trainer does not support decoding ids.")

    @abstractmethod
    def encode(self, inputs, labels, tokenizer) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pass

    def update_metrics(self, metrics: Metrics, batch, outputs):
        metrics.update(batch, outputs)

    @abstractmethod
    def forward_to_model(self, model, criterion, batch):
        pass

    @abstractmethod
    def generate(self, model, batch):
        pass

    def _process_data(self, tokenizer, data, sort_by_bracket_depth, set_size):
        if not sort_by_bracket_depth:
            data.sort(key=(lambda entry: len(entry.get('i'))))  # sort by input length
        else:
            data.sort(key=(lambda entry: get_bracket_depth(entry.get('i'))))

        # Log data metrics
        print(f"Dataset size: {len(data)}")
        print(f"Dataset input length span: [{len(data[0].get('i'))}, {len(data[set_size - 1].get('i'))}]")
        sequence_lengths = [[x, len(entry.get('i'))] for (x, entry) in enumerate(data)]

        table = wandb.Table(
            data=sequence_lengths[::int(set_size / 100)],
            columns=["x", "y"]
        )
        wandb.log({
            "sequence_lengths": wandb.plot.line(table, "x", "y", title="Input Sequence Lengths")
        })

        # encode inputs & labels
        inputs = [entry.get("i") for entry in data]
        labels = [entry.get("l") for entry in data]
        input_ids, input_masks, label_ids, label_masks = self.encode(inputs=inputs, labels=labels, tokenizer=tokenizer)

        # Log Encoded lengths
        lengths = list(map(lambda x: torch.count_nonzero(x).item(), input_masks))
        print(f"Tokenized input length span: [{min(lengths)}, {max(lengths)}]")
        encoded_lengths = [[x, lengths[x]] for x in range(len(lengths))]
        table = wandb.Table(
            data=encoded_lengths[::int(set_size / 100)],
            columns=["x", "y"]
        )
        wandb.log({
            "encoded_lengths": wandb.plot.line(table, "x", "y", title="Tokenized Sequence Lengths")
        })

        # Sort data
        return tuple(sort_by_length(input_ids, input_masks, label_ids, label_masks))

    def split_and_prepare(self, tokenizer, data, tokenizer_name, sort_by_bracket_depth=False, cap_size=-1):
        # Sort by sequence length, log lengths
        set_size = len(data)
        if not cap_size >= set_size and cap_size != -1:
            data = random.sample(data, cap_size)
            set_size = len(data)

        from_file_buffer = Common.load_tensor_data(tokenizer_name)
        if not from_file_buffer:
            print(f"No preprocessed data found for {tokenizer_name}. Preprocessing data instead.")
            data = self._process_data(tokenizer, data, sort_by_bracket_depth, set_size)
            print(f"Preprocessing done. Saving data locally.")
            Common.save_tensor_data(tokenizer_name, data)
        else:
            data = from_file_buffer
        input_ids, input_masks, label_ids, label_masks = data

        # Split ids and masks for inputs and labels into test and training sets, respectively
        train = {}
        test = {}
        train['in_ids'], test['in_ids'], train['in_masks'], test['in_masks'], \
        train['l_ids'], test['l_ids'], train['l_masks'], test['l_masks'] = \
            train_test_split(input_ids, input_masks, label_ids, label_masks, test_size=0.3, random_state=420,
                             shuffle=False)

        train_set = TensorDataset(*sort_by_length(*itemgetter('in_ids', 'in_masks', 'l_ids', 'l_masks')(train)))
        test_set = TensorDataset(*sort_by_length(*itemgetter('in_ids', 'in_masks', 'l_ids', 'l_masks')(test)))
        return train_set, test_set

    def train(self, model, optimizer, tokenizer, criterion, data_set, metrics: Metrics):
        device = self.hp.device
        num_epochs = self.hp.num_epochs
        data_loader = DataLoader(data_set, batch_size=self.hp.batch_size, shuffle=True, drop_last=True)
        data_set_size = len(data_loader)
        print_every = self.hp.print_every if self.hp.print_every else int(data_set_size / 10)
        print_every = max(print_every, 1)


        print(">> Training Started <<")
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            metrics.reset()

            print("Starting Epoch: {}/{}".format(epoch + 1, num_epochs))
            for i, batch in enumerate(data_loader):
                # Move all tensors to the device
                batch = tuple(tensor.to(device) for tensor in batch)

                # forward
                optimizer.zero_grad()
                outputs, loss = self.forward_to_model(model=model, criterion=criterion, batch=batch)

                # Update metrics
                total_loss += loss.item()
                self.update_metrics(metrics, batch, outputs)

                # back propagate
                loss.backward()

                # Apply gradient clipping
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.hp.max_norm) # Disabled gradient clipping

                # update optimizer
                optimizer.step()

                # print info
                if (i + 1) % print_every == 0 or (i + 1) == data_set_size:
                    # for last batch in epoch use residual, else use hp.print_every
                    count = print_every if (i + 1) % print_every == 0 \
                        else data_set_size % print_every

                    prefix = ""
                    metrics.print(index=i + 1, set_size=data_set_size, prefix=prefix, count=count, epoch=epoch,
                                  loss=total_loss, decoder=lambda x: self.decode_tokens(tokenizer, x))


                    total_loss = 0
                    metrics.reset()
                    if device == "cuda":
                        torch.cuda.empty_cache()

    def test(self, model, optimizer, tokenizer, criterion, data_set, metrics: Metrics, split_into_two=False, generate=False, log_prefix=None, pad_output_to=-1):
        device = self.hp.device

        data_loader = DataLoader(data_set, batch_size=self.hp.batch_size, shuffle=False, drop_last=True)
        data_set_size = len(data_loader)
        print_every = self.hp.print_every if self.hp.print_every else int(data_set_size / 100)
        print_every = max(print_every, 1)
        metrics.reset()

        print(">> Evaluation Started <<")

        with torch.no_grad():
            model.eval()
            total_loss = 0

            for i, batch in enumerate(data_loader):
                # Move all tensors to the device
                batch = tuple(tensor.to(device) for tensor in batch)

                # forward
                optimizer.zero_grad()

                if generate:
                    outputs = self.generate(model, batch)
                    diff = pad_output_to - outputs.size(1)
                    if pad_output_to > 0 and diff > 0:
                        bs = outputs.size(0)
                        pad_token = tokenizer.pad_token_id
                        pads = torch.full((bs, diff), pad_token, device=outputs.device, dtype=outputs.dtype)
                        outputs = torch.cat((outputs, pads), dim=1)
                    elif diff < 0:
                        outputs = outputs[:, :diff]
                    outputs = torch.nn.functional.one_hot(outputs, num_classes=len(tokenizer))
                    loss = torch.tensor(0)
                else:
                    outputs, loss = self.forward_to_model(model=model, criterion=criterion, batch=batch)

                # Update metrics
                total_loss += loss.item()
                self.update_metrics(metrics, batch, outputs)

                # print info
                do_print = (i + 1) % print_every == 0 if not split_into_two else i + 1 == int(data_set_size / 2)
                if do_print or (i + 1) == data_set_size:
                    # for last batch in epoch use residual, else use hp.print_every
                    count = \
                        data_set_size / 2 if split_into_two \
                            else print_every if (i + 1) % print_every == 0 \
                            else data_set_size % print_every

                    bin_index = 1 if i < (data_set_size / 2) else 2
                    prefix = "eval_" if not split_into_two else "bin_"
                    prefix = prefix if not log_prefix else log_prefix
                    metrics.print(index=i + 1, set_size=data_set_size, prefix=prefix, count=count, epoch=0,
                                  loss=total_loss, decoder=lambda x: self.decode_tokens(tokenizer, x))


                    total_loss = 0
                    metrics.reset()
                    if device == "cuda":
                        torch.cuda.empty_cache()

