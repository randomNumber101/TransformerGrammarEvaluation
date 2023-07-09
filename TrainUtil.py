from abc import ABC, abstractmethod
import os.path
import random
import re
import sys
import argparse

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

    def split_and_prepare(self, tokenizer, data, split_test=False):
        # Sort by sequence length, log lengths
        data.sort(key=(lambda entry: len(entry.get('i'))))  # sort by input length
        set_size = len(data)

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
        input_ids, input_masks, label_ids, label_masks = sort_by_length(input_ids, input_masks, label_ids, label_masks)

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
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.hp.max_norm)
                # update optimizer
                optimizer.step()

                # print info
                if (i + 1) % self.hp.print_every == 0 or (i + 1) == data_set_size:
                    # for last batch in epoch use residual, else use hp.print_every
                    count = self.hp.print_every if (i + 1) % self.hp.print_every == 0 \
                        else data_set_size % self.hp.print_every

                    prefix = ""
                    metrics.print(index=i + 1, set_size=data_set_size, prefix=prefix, count=count, epoch=epoch,
                                  loss=total_loss, decoder=lambda x: self.decode_tokens(tokenizer, x))


                    total_loss = 0
                    metrics.reset()
                    if device == "cuda":
                        torch.cuda.empty_cache()

    def eval(self, model, optimizer, tokenizer, criterion, data_set, metrics: Metrics, split_into_two=False, log_prefix=None):
        device = self.hp.device
        print_every = int(self.hp.print_every / 3)
        data_loader = DataLoader(data_set, batch_size=self.hp.batch_size, shuffle=False, drop_last=True)
        data_set_size = len(data_loader)
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
                    prefix = "eval_" if not split_into_two else "bin_" + str(bin_index) + "_"
                    prefix = prefix if not log_prefix else log_prefix
                    metrics.print(index=i + 1, set_size=data_set_size, prefix=prefix, count=count, epoch=0,
                                  loss=total_loss, decoder=lambda x: self.decode_tokens(tokenizer, x))




                    total_loss = 0
                    metrics.reset()
                    if device == "cuda":
                        torch.cuda.empty_cache()
