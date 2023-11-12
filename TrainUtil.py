import gc
from abc import ABC, abstractmethod
import os.path
import random
from typing import Union, Sized

import torch
from torch import Tensor
from torch.cuda.amp import GradScaler
from torch.optim import Optimizer

import wandb
from sklearn.model_selection import train_test_split

from torch.utils.data.dataset import Dataset, T_co
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Sampler

from operator import itemgetter
import numpy as np
from random import shuffle
import Common

import time

class Timer:

    disabled = []
    timers = {}

    @staticmethod
    def disable(name):
        Timer.disabled.append(name)

    @staticmethod
    def start(name):
        Timer.timers[name] = time.time()

    @staticmethod
    def measure(name, avg_over=1, tag=None):
        now = time.time()
        if name not in Timer.disabled:
            prev = Timer.timers[name]
            delta = (now - prev) / avg_over
            print(f"{name} time delta: {str(delta)} - {tag if tag else 'Total'}")
        Timer.timers[name] = now


# This code is from https://colab.research.google.com/github/jaygala24/pytorch-implementations/blob/master/Attention%20Is%20All%20You%20Need.ipynb#scrollTo=TWOqbtFNzcU_
class NoamOptim(object):
    """ Optimizer wrapper for learning rate scheduling.
    """

    def __init__(self, optimizer: Optimizer, d_model, factor=2, n_warmup_steps=4000):
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

    def state_dict(self):
        return self.optimizer.state_dict()

    def add_param_group(self, param_group):
        return self.optimizer.add_param_group(param_group)

    def load_state_dict(self, arg):
        self.optimizer.load_state_dict(arg)

    def __getattr__(self, name):
        # Redirect attribute access to the original object
        return getattr(self.optimizer, name)


def get_attended_token_count(mask):
    # expecting shape (batch_size, seq_length)
    return torch.count_nonzero(mask, dim=1)


def pick_all(data, entry):
    return list(map(lambda t: t[entry], data))


def sort_by_length(a, masks, c, d, return_indices=False):
    indices = list(range(len(a)))
    sorted_tuples = sorted(zip(a, masks, c, d, indices),
                           key=lambda x: torch.count_nonzero(
                               x[1]))  # sort by number of non-zero entries in the input-masks
    sorted_tensors = tuple(torch.stack(pick_all(sorted_tuples, i)) for i in range(4))
    indices = pick_all(sorted_tuples, 4)
    if return_indices:
        return sorted_tensors, indices
    return sorted_tensors


def take_n(a, m, c, d, n):
    out = list(zip(a, m, c, d))[-n:]
    return tuple(torch.stack(pick_all(out, i)) for i in range(4))


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


class TensorListDataset(Dataset):
    def __init__(self, tensors: TensorDataset, list: list[object]):
        self.tensors = tensors
        self.list = list

    def __getitem__(self, index) -> T_co:
        return self.tensors[index], self.list[index]

    def __len__(self):
        return len(self.tensors)


from random import shuffle


class BucketLoader(Sized):
    def __init__(self, data_source: Dataset, bucket_boundaries, batch_sizes=None, with_stats=False):
        self.data_source = data_source
        self.with_stats = with_stats
        ind_n_len = []
        for i, p in enumerate(data_source):
            ind_n_len.append((i, self.get_len(p, with_stats)))
        self.ind_n_len = ind_n_len
        self.bucket_boundaries = bucket_boundaries
        self.batch_sizes = dict((bucket_boundaries[i], batch_sizes[i]) for i in range(len(bucket_boundaries)))

        data_buckets = {}
        # where p is the id number and seq_len is the length of this id number.
        for indx, seq_len in self.ind_n_len:
            pid = self.element_to_bucket_id(indx, seq_len)
            p = self.unpad_pack(self.data_source[indx], pid, self.with_stats)
            if pid in data_buckets.keys():
                data_buckets[pid].append((indx, pid))
            else:
                data_buckets[pid] = [(indx, pid)]

        for k, l in data_buckets.items():
            shuffle(l)
            batch_size = self.batch_sizes[k]
            data_buckets[k] = [l[i:i + batch_size] for i in range(0, len(l), batch_size)]
            print(f"{len(data_buckets[k])} batches of size {batch_size} for sequence length {k}.")

        self.data_buckets = data_buckets


    def get_len(self, data_point, with_stats=False):
        if with_stats:
            return self.get_len(data_point[0], with_stats=False)
        return max(
            torch.count_nonzero(data_point[1]).item(),  # Count inputs
            torch.count_nonzero(data_point[3]).item())  # Count labels

    def unpad_pack(self, data_point, length, with_stats=False):
        if with_stats:
            return self.unpad_pack(data_point[0], length), data_point[1]
        return tuple(t[:length] if t.numel() > length else t for t in data_point)

    def _pre_collate(self, data, with_stats=False):
        if not with_stats:
            return tuple(torch.stack(list(x)) for x in zip(*data))
        else:
            stats = [t[-1] for t in data]
            tensors = [[t[0] for t in data]]
            return self._pre_collate(tensors), torch.utils.data.default_collate(stats)

    def __iter__(self):
        iter_list = []
        for k in self.data_buckets.keys():
            iter_list.extend(self.data_buckets[k])
        shuffle(iter_list)  # shuffle all the batches so they aren't ordered by bucket size
        for i in iter_list:
            def map_idx(indx, pid):
                return self.unpad_pack(self.data_source[indx], pid, self.with_stats)
            items = [map_idx(i, p) for (i, p) in i]
            yield torch.utils.data.default_collate(items)  # as it was stored in an array

    def __len__(self):
        length = 0
        for (k, l) in self.data_buckets.items():
            length += len(l)
        return length

    def element_to_bucket_id(self, x, seq_length):
        for i, b in enumerate(self.bucket_boundaries):
            if seq_length < b:
                return b
        return self.bucket_boundaries[len(self.bucket_boundaries) - 1]


class Trainer(ABC):
    def __init__(self, hp):
        self.hp = hp
        self.save_dir = "."
        self.best_loss = float("inf")

    def decode_tokens(self, tokenizer, ids) -> list[str]:
        raise NotImplementedError("This trainer does not support decoding ids.")

    def save(self, model, loss, optimizer=None):
        save_dir = os.path.join(self.save_dir, "checkpoints")
        if loss < self.best_loss:
            Common.save(save_dir, "checkpoint_best", model)
            self.best_loss = loss
        else:
            Common.save(save_dir, "checkpoint_last", model, optimizer)

    def load_state_dict(self):
        save_dir = os.path.join(self.save_dir, "checkpoints")
        best = Common.get_state_dict("checkpoint_best", path=save_dir)
        if best:
            print("Loaded best model state dict.")
            return best
        return Common.get_state_dict("checkpoint_last", path=save_dir)

    @abstractmethod
    def encode(self, inputs, labels, tokenizer, return_entry_ids=False) -> Union[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int]]]:
        pass

    def update_metrics(self, metrics: Metrics, batch, outputs):
        metrics.update(batch, outputs.to(torch.float32))

    @abstractmethod
    def forward_to_model(self, model, criterion, batch):
        pass

    @abstractmethod
    def generate(self, model, batch):
        pass

    def _process_data(self, tokenizer, data, return_stats=False, cap_size=-1, take_longest=True):
        # Sort by sequence length, log lengths
        data.sort(key=(lambda entry: len(entry.get('i'))))  # sort by input length
        set_size = len(data)
        if not cap_size >= set_size and cap_size != -1:
            if not take_longest:
                data = random.sample(data, cap_size)
            else:
                data = data[-cap_size:]
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

        entry_ids, input_ids, input_masks, label_ids, label_masks = self.encode(inputs=inputs, labels=labels,
                                                                                tokenizer=tokenizer,
                                                                                return_entry_ids=True)

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
        if not return_stats:
            return tuple(sort_by_length(input_ids, input_masks, label_ids, label_masks))
        else:
            stats = [entry.get("stats") for entry in data]
            stats = [stats[i] for i in entry_ids]
            return (input_ids, input_masks, label_ids, label_masks), stats

    def split_and_prepare(self, tokenizer, data, tokenizer_name, cap_size=-1, test_portion=0.3, train_portion=None,
                        take_longest=False, force_encoding=False):

        return_stats = "stats" in data[0]



        from_file_buffer = Common.load_tensor_data(tokenizer_name, return_stats=return_stats)
        if (not from_file_buffer) or force_encoding:
            print(f"No preprocessed data found for {tokenizer_name}. Preprocessing data instead.")
            data = self._process_data(tokenizer, data, return_stats, cap_size, take_longest)
            print(f"Preprocessing done. Saving data locally.")
            if return_stats:
                data, stats = data
                if not force_encoding:
                    Common.save_tensor_data(tokenizer_name, data, stats)
            else:
                if not force_encoding:
                    Common.save_tensor_data(tokenizer_name, data)
        else:
            data = from_file_buffer
            if return_stats:
                data, stats = data
            set_size = len(data[0])
            if not cap_size >= set_size and cap_size != -1:
                sample_indices = random.sample(range(set_size), cap_size)
                sample_indices.sort()
                data = (tensor[sample_indices] for tensor in data)
                if return_stats:
                    stats = [stats[i] for i in sample_indices]

        num_tokens = self.hp.input_size
        input_ids, input_masks, label_ids, label_masks = tuple(t[:, :num_tokens] for t in data)


        if not return_stats:
            # Split ids and masks for inputs and labels into test and training sets, respectively. Sorth by length.
            train = {}
            test = {}
            train['in_ids'], test['in_ids'], train['in_masks'], test['in_masks'], \
            train['l_ids'], test['l_ids'], train['l_masks'], test['l_masks'] = \
                train_test_split(input_ids, input_masks, label_ids, label_masks, test_size=test_portion,
                                 random_state=420, shuffle=False)
            train_set = TensorDataset(*sort_by_length(*itemgetter('in_ids', 'in_masks', 'l_ids', 'l_masks')(train)))
            test_set = TensorDataset(*sort_by_length(*itemgetter('in_ids', 'in_masks', 'l_ids', 'l_masks')(test)))

            return train_set, test_set

        else:
            # Return stats too.
            train = {}
            test = {}
            train['in_ids'], test['in_ids'], train['in_masks'], test['in_masks'], \
            train['l_ids'], test['l_ids'], train['l_masks'], test['l_masks'], \
            train_stats, test_stats = \
                train_test_split(input_ids, input_masks, label_ids, label_masks, stats, test_size=test_portion,
                                 random_state=420, shuffle=False)

            train_tuples, train_indices = sort_by_length(*itemgetter('in_ids', 'in_masks', 'l_ids', 'l_masks')(train),
                                                         True)
            test_tuples, test_indices = sort_by_length(*itemgetter('in_ids', 'in_masks', 'l_ids', 'l_masks')(test),
                                                       True)

            if train_portion:
                train_size = int(len(train_tuples[0]) * train_portion)
                train_tuples = take_n(*train_tuples, train_size)
                train_indices = train_indices[-train_size:]

            train_set = TensorDataset(*train_tuples)
            test_set = TensorDataset(*test_tuples)

            train_stats = [train_stats[i] for i in train_indices]
            test_stats = [test_stats[i] for i in test_indices]

            return TensorListDataset(train_set, train_stats), TensorListDataset(test_set, test_stats)

    def train(self, model, optimizer, tokenizer, criterion, data_set: torch.utils.data.Dataset, metrics: Metrics):
        device = self.hp.device
        num_epochs = self.hp.num_epochs  #
        has_additional_info = True if isinstance(data_set, TensorListDataset) else False

        if self.hp.dynamic_batching:
            data_loader = BucketLoader(data_set, [64, 128, 256, 512, 1024], batch_sizes=[16, 8, 4, 4, 2],
                                       with_stats=has_additional_info)
        else:
            data_loader = DataLoader(data_set, batch_size=self.hp.batch_size, shuffle=True,
                                     drop_last=True)  # No bucketing

        # data_loader = DataLoader(data_set, batch_size=1, shuffle=False, drop_last=False, sampler=bucket_sampler, pin_memory=False)
        data_set_size = len(data_loader)
        print_every = self.hp.print_every if self.hp.print_every else int(data_set_size / 10)
        print_every = max(print_every, 1)
        save_every = max(int(data_set_size / 3), 100)

        Timer.disable("Batch-wise")

        # Memory Optimization

        scaler = GradScaler()

        def free_memory(batch):
            for t in (batch if has_additional_info else batch[0]):
                del t
            gc.collect()
            torch.cuda.empty_cache()

        print(">> Training Started <<")

        Timer.start("Per-batch")


        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            metrics.reset()

            print("Starting Epoch: {}/{}".format(epoch + 1, num_epochs))
            for i, batch in enumerate(data_loader):

                Timer.start("Batch-wise")

                # Move all tensors to the device
                if has_additional_info:
                    batch = tuple(tensor.to(device) for tensor in batch[0]), batch[1]
                else:
                    batch = tuple(tensor.to(device) for tensor in batch)

                # forward
                optimizer.zero_grad()
                tensor_batch = batch if not has_additional_info else batch[0]

                with torch.autocast(device_type=device):
                    outputs, loss = self.forward_to_model(model=model, criterion=criterion, batch=tensor_batch)

                Timer.measure("Batch-wise", tag="Forward")

                # Update metrics
                total_loss += loss.item()
                self.update_metrics(metrics, batch, outputs)

                # back propagate
                scaler.scale(loss).backward()

                # Apply gradient clipping
                if self.hp.max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.hp.max_norm)
                Timer.measure("Batch-wise", tag="Clip")

                # Optimizer step
                scaler.unscale_(optimizer)  # Unscale for gradient clipping
                Timer.measure("Batch-wise", tag="Backward")

                # update scaler & optimizer
                scaler.step(optimizer)
                scaler.update()
                Timer.measure("Batch-wise", tag="Optimizer")

                # print info
                if (i + 1) % print_every == 0 or (i + 1) == data_set_size:
                    # for last batch in epoch use residual, else use hp.print_every
                    count = print_every if (i + 1) % print_every == 0 \
                        else data_set_size % print_every

                    prefix = ""
                    metrics.print(index=i + 1, set_size=data_set_size, prefix=prefix, count=count, epoch=epoch,
                                  loss=total_loss, decoder=lambda x: self.decode_tokens(tokenizer, x))

                    print(f"Learning Rate: {optimizer.param_groups[0]['lr']}")
                    total_loss = 0
                    metrics.reset()

                if (i + 1) % save_every == 0 or (i + 1) == data_set_size:
                    self.save(model, loss.item(), optimizer)

                if i > 0 and i % 25 == 0:
                    Timer.measure("Per-batch", avg_over=25)

                # free_memory(batch)

                Timer.measure("Batch-wise", tag="Memory-freeing")

    def test(self, model, optimizer, tokenizer, criterion, data_set, metrics: Metrics, split_into_two=False,
             generate=False, log_prefix=None, pad_output_to=-1):
        device = self.hp.device
        has_additional_info = True if isinstance(data_set, TensorListDataset) else False
        data_loader = DataLoader(data_set, batch_size=self.hp.batch_size, shuffle=False, drop_last=True)
        data_set_size = len(data_loader)
        print_every = self.hp.print_every if self.hp.print_every else int(data_set_size / 100)
        print_every = max(print_every, 1)
        if has_additional_info:
            print_every = 1
        metrics.reset()


        print(">> Evaluation Started <<")
        with torch.no_grad():
            model.eval()
            total_loss = 0

            for i, batch in enumerate(data_loader):
                # Move all tensors to the device
                if has_additional_info:
                    batch = tuple(tensor.to(device) for tensor in batch[0]), batch[1]
                else:
                    batch = tuple(tensor.to(device) for tensor in batch)
                tensor_batch = batch if not has_additional_info else batch[0]
                # forward
                optimizer.zero_grad()

                if generate:
                    outputs = self.generate(model, tensor_batch)
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
                    outputs, loss = self.forward_to_model(model=model, criterion=criterion, batch=tensor_batch)

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
