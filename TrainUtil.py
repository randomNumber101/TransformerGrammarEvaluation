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


class Metrics(ABC):
    @abstractmethod
    def update(self, batch, outputs):
        pass

    @abstractmethod
    def print(self, index, set_size, prefix, count, epoch, loss):
        pass

    @abstractmethod
    def reset(self):
        pass


class Trainer(ABC):
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-d', '--set_name', type=str, help="The data set", required=True)
        parser.add_argument('-bs', '--batch_size', type=int, help="Batch size")
        parser.add_argument('-is', '--input_size', type=int, help="Input (and output) Length of the network")
        parser.add_argument('-epochs', '--num_epochs', type=int, help="Epoch count")
        parser.add_argument('-lr', '--learning_rate', type=float, help="Learning rate")
        parser.add_argument('-mn', '--max_norm', type=float, help="Maximal gradient for gradient clipping")
        parser.add_argument('-pe', '--print_every', type=int, help="Frequency of logging results")

        args = parser.parse_args()

        # Set script Hyper Parameters
        hp = Common.HyperParams()
        Common.overwrite_params(hp, vars(args))
        hp.device = "cuda" if torch.cuda.is_available() else "cpu"  # Check if CUDA is available

        print(f"hyper parameters: {vars(hp)}")
        self.hp = hp
        self.args = args

    @abstractmethod
    def encode(self, inputs, labels, tokenizer):
        pass

    @abstractmethod
    def forward_to_model(self, model, criterion, batch):
        pass

    def split_and_prepare(self, tokenizer, data):
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

        # Split ids and masks for inputs and labels into test and training sets, respectively
        train = {}
        test = {}
        train['in_ids'], test['in_ids'], train['in_masks'], test['in_masks'], \
        train['l_ids'], test['l_ids'], train['l_masks'], test['l_masks'] = \
            train_test_split(input_ids, input_masks, label_ids, label_masks, test_size=0.2, random_state=420)

        train_set = TensorDataset(*itemgetter('in_ids', 'in_masks', 'l_ids', 'l_masks')(train))
        train['loader'] = DataLoader(train_set, batch_size=self.hp.batch_size, shuffle=True, drop_last=True)

        test_set = TensorDataset(*itemgetter('in_ids', 'in_masks', 'l_ids', 'l_masks')(test))
        test['loader'] = DataLoader(test_set, batch_size=self.hp.batch_size, shuffle=True, drop_last=True)
        return train, test

    def train(self, model, optimizer, criterion, data_set, metrics: Metrics, is_training=True):
        device = self.hp.device
        num_epochs = self.hp.num_epochs if is_training else 1
        data_set_size = len(data_set["loader"])

        for epoch in range(num_epochs):
            model.train() if is_training else model.eval()
            total_loss = 0

            print("Starting Epoch: {}/{}".format(epoch + 1, num_epochs))

            for i, batch in enumerate(data_set['loader']):
                # Move all tensors to the device
                batch = tuple(tensor.to(device) for tensor in batch)

                # forward
                optimizer.zero_grad()
                outputs, loss = self.forward_to_model(model=model, criterion=criterion, batch=batch)

                # Update metrics
                total_loss += loss.item()
                metrics.update(batch=batch, outputs=outputs)

                if is_training:
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

                    prefix = "" if is_training else "eval_"
                    metrics.print(index=i + 1, set_size=data_set_size, prefix=prefix, count=count, epoch=epoch,
                                  loss=total_loss)

                    total_loss = 0
                    metrics.reset()
                    if device == "cuda":
                        torch.cuda.empty_cache()
