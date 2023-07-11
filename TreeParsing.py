"""
    In:
        He goes to the mall.
    Out:
        ((He))((goes)((to)((the)(mall))))
"""
import datetime
import os
import random
import re
from enum import Enum
from typing import List, Union

import textdistance
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.nn.functional import one_hot
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import TensorDataset

import BaseLines
import TrainUtil
import wandb
import numpy as np

from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig
from torch.optim import AdamW

from BaseLines import TokenEmbedder
from Common import save
from DataUtil import load_data
import Common

'''
hp.num_epochs = 5
hp.batch_size = 32
hp.print_every = 10
hp.max_length = -1
hp.max_norm = 5.0
'''

EPS = 0.05  ## Error EPS
bracket_pattern = r"\(\w*|\)\w*"


def start():
    args, hp = Common.loadParams()
    model_name = args.model
    model_postfix = "SMALL" if args.layers > 0 else "PRETRAINED"

    # Load Training Data
    training_set_name = "Grammar1_7143222753796263824.json" if not args.set_name else args.set_name
    dataset = load_data(Common.training_folder + training_set_name)  # Load data from file
    print("Loaded dataset: " + training_set_name)

    # Initialize Tokenizer
    if args.tokenize == "words":
        tokenizer = BracketTokenizer.from_pretrained('facebook/bart-base').word_wise()
    elif args.tokenize == "words_bpe":
        tokenizer = BracketTokenizer.from_pretrained('facebook/bart-base')  # Use custom tokenizer
    else:
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')  # Use default tokenizer

    if model_name == "BART":
        trainer = TransductionTrainer(hp)
    elif model_name == "LSTM":
        trainer = Seq2SeqLSTMTrainer(hp, tokenizer)

    # Initialize weights & biases for logging
    config = {
        "model": "BartForConditionalGeneration" if model_name == "BART" else model_name,
        "optimizer": "AdamW",
        "criterion": "CrossEntropy",
        "training_set": training_set_name,
        "training_set_size": len(dataset['data']),

        "batch_size": hp.batch_size,
        "input_size": hp.input_size,
        "batch_eval_count": hp.print_every,

        "num_epochs": hp.num_epochs,
        "learning_rate": hp.learning_rate,
        "gradient_clipping_max": hp.max_norm,

        "num_layers": args.layers
    }
    tags = [
        "TreeBracketing",
        training_set_name.replace(".json", ""),
        model_name,
        model_postfix
    ]
    wandb_mode = 'disabled' if args.test_mode else 'online'
    wandb.init(project="Tree Bracketing", config=config, tags=tags, mode=wandb_mode)
    wandb.define_metric("loss", summary='min')
    wandb.define_metric("accuracy", summary='max')

    # Split and prepare training data
    train_set, test_set = trainer.split_and_prepare(tokenizer, dataset['data'])
    # train_set = TensorDataset(*train_set[0:int(len(train_set) / 2)])

    # Define Model
    if model_name == "BART":
        config = BartConfig.from_pretrained('facebook/bart-base')
        if args.layers > 0:
            config.num_hidden_layers = args.layers
        model = BartForConditionalGeneration.from_pretrained('facebook/bart-base',
                                                             config=config)  # BART Decoder-Encoder for Seq2Seq
        model.resize_token_embeddings(len(tokenizer))
        metrics = TransductionMetrics()
    else:
        model = BaseLines.Seq2SeqBiLSTM(vocab_size=tokenizer.vocab_size, input_dim=128, hidden_dim=1024)
        metrics = LSTMTransductionMetrics(model)
    model.to(device=hp.device)

    # Initialize optimizer and criterion
    optimizer = AdamW(model.parameters(), lr=hp.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    criterion.to(hp.device)

    # Start training
    trainer.train(model=model, optimizer=optimizer, tokenizer=tokenizer, criterion=criterion, data_set=train_set,
                  metrics=metrics)

    # Evaluate
    trainer.eval(model=model, optimizer=optimizer, tokenizer=tokenizer, criterion=criterion, data_set=test_set,
                 metrics=metrics,
                 split_into_two=False)

    # Task specific evaluation
    # trainer.eval_labels()

    # Save Model & Optimizer
    name = "BRACKETING_" + training_set_name.replace(".json", "") + model_name + "_" + model_postfix
    save(name, model=model, optimizer=optimizer)


class BracketTokenizer(BartTokenizer):
    def __init__(self, *args, **kwargs):
        super(BracketTokenizer, self).__init__(*args, **kwargs, add_prefix_space=True)
        # Regex: Either open bracket with a type (word), closed bracket or a string not containing brackets or space.
        self.token_pattern = r"\(\w*|\)|[^()\s]+|\s+"
        self.bracket_ids = {}
        self.next_bracket_id_index = 0
        self.bracket_types = set()
        self.do_word_wise = False
        self.grammar_vocab = set()

    def word_wise(self):
        self.encoder = {
            "<s>": 0,
            "<pad>": 1,
            "</s>": 2,
            "<unk>": 3,
            "<mask>": 4
        }
        self.decoder = {}
        self.do_word_wise = True
        return self

    def get_bracket_token(self, bracket_type, opening):
        t = None
        if not opening:
            t = ')' + bracket_type
        else:
            t = '(' + bracket_type

        if t not in self.bracket_types:
            self.bracket_types.add(t)
            self.add_tokens(t)

        return t

    def tokenize(self, text, **kwargs):
        basicTokens = re.findall(pattern=self.token_pattern, string=text)
        split_tokens = []
        bracket_stack = []
        for token in basicTokens:
            if token.startswith("("):
                bracket_type = token[1:] if len(token) > 1 else "-NONE-"
                bracket_stack.append(bracket_type)
                token = self.get_bracket_token(bracket_type=bracket_type, opening=True)
            elif token == ')' and bracket_stack:
                bracket_type = bracket_stack.pop()
                token = self.get_bracket_token(bracket_type=bracket_type, opening=False)
            if self.do_word_wise:
                if token not in self.grammar_vocab:
                    self.grammar_vocab.add(token)
                    self.add_tokens(token)
                split_tokens.append(token)
            else:
                split_tokens.extend(super(BracketTokenizer, self)._tokenize(token))
        return split_tokens


def calculate_edit_distance(labels, predicted):
    index = random.randrange(labels.size(dim=0))
    return textdistance.levenshtein.distance(labels[index, :].tolist(), predicted[index, :].tolist()) * labels.size(
        dim=0) / labels.size(dim=1)


def brackets_only(s: str):
    return re.findall(bracket_pattern, s)


def bracket_edit_distance(label, prediction):
    return textdistance.levenshtein.distance(brackets_only(label), brackets_only(prediction))


def bracket_accuracy(label, prediction):
    label = brackets_only(label)
    prediction = brackets_only(prediction)

    if len(prediction) < len(label):
        for i in range(len(label) - len(prediction)):
            prediction.append("<pad>")
    prediction = prediction[0: len(label)]
    return accuracy_score(label, prediction)


def calculate_edit_distance_all(labels, predicted):
    pairs = zip(labels.tolist(), predicted.tolist())
    return np.sum([textdistance.levenshtein.distance(entry[0], entry[1]) for entry in pairs])


def count_full_hit_percentage(labels, predicted):
    non_full_hits = torch.count_nonzero(torch.count_nonzero(labels - predicted, dim=1))
    num_full_hits = labels.size(0) - non_full_hits
    return num_full_hits.item() / labels.size(0)


class TransductionMetrics(TrainUtil.Metrics):

    def __init__(self):
        super().__init__()
        self.metrics = np.zeros(5)
        self.errors = []

    def update(self, batch, outputs):
        input_ids, in_masks, labels, _ = (tensor.cpu().detach() for tensor in batch)
        outputs = outputs.cpu().detach()

        # Adapt
        labels = labels[:, 1:]
        predicted_indices = torch.argmax(outputs, dim=-1)

        # Sample one erroneous prediction
        error_indices = torch.nonzero(torch.any(predicted_indices != labels, dim=1)).squeeze().squeeze().tolist()
        if error_indices:
            if not isinstance(error_indices, list):
                error_indices = [error_indices]
            idx = error_indices[random.randrange(len(error_indices))]
            self.errors.append((input_ids[idx], labels[idx], predicted_indices[idx]))

        assert labels.size() == predicted_indices.size(), f"{labels.size()} != {predicted_indices.size()}"

        # Calculate metrics
        avg_length = torch.count_nonzero(in_masks) / in_masks.size(0)
        full_hit_perc = count_full_hit_percentage(labels=labels, predicted=predicted_indices.reshape(labels.size()))  # full hits

        # flatten
        labels = labels.reshape(-1, 1)
        predicted_indices = predicted_indices.reshape(-1, 1)

        # Calculate residual
        acc = accuracy_score(y_true=labels, y_pred=predicted_indices)  # accuracy,
        f1 = f1_score(y_true=labels, y_pred=predicted_indices, average="micro")  # f1
        edit_dist = calculate_edit_distance(labels, predicted_indices)  # levenshtein

        # Update
        self.metrics += np.array([avg_length, acc, f1, edit_dist, full_hit_perc])

    def print(self, index, set_size, prefix, count, epoch, loss, decoder=None):
        avg_loss = loss / count
        metrics = self.metrics / count
        [avg_length, accuracy, f1, levenshtein, full_hit_perc] = metrics
        print(
            f"Batch {index}/{set_size} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Lengths : {avg_length:.2f}")

        error = self.errors[random.randrange(len(self.errors))]
        inp, label, pred = (decoder(ids) for ids in error)
        print(f"IN: \n\t{inp} \nOUT: \n\t{pred} \nTRUE: \n\t{label}\n")

        bracket_acc = 0.
        for error in self.errors:
            inp, label, pred = (decoder(ids) for ids in error)
            bracket_acc += bracket_accuracy(label, pred)

        bracket_acc /= len(self.errors)

        wandb.log({
            prefix + "in_lengths": avg_length,
            prefix + "epoch": epoch,
            prefix + "loss": avg_loss,
            prefix + "accuracy": accuracy,
            prefix + "f1": f1,
            prefix + "levenshtein": levenshtein,
            prefix + "full_hit%": full_hit_perc,
            prefix + "bracket_acc": bracket_acc
        })

    def reset(self):
        self.metrics = np.zeros(5)
        self.errors = []


class TransductionTrainer(TrainUtil.Trainer):
    def __init__(self, hp):
        super().__init__(hp)

    def decode_tokens(self, tokenizer: BartTokenizer, ids) -> str | list[str]:
        if len(ids.size()) == 1:
            return tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return [tokenizer.decode(ids[i], skip_special_tokens=True, clean_up_tokenization_spaces=True) for i in
                range(ids.size(0))]

    def encode(self, inputs, labels, tokenizer):
        in_ids = []
        in_masks = []
        l_ids = []
        l_masks = []

        filtered_count = 0
        max_length = self.hp.input_size if self.hp.input_size > 0 else tokenizer.model_max_length

        for (i, l) in zip(inputs, labels):
            encoded_i = tokenizer.encode_plus(i, add_special_tokens=True, padding='max_length',
                                              max_length=max_length,
                                              return_attention_mask=True, return_tensors='pt')

            encoded_l = tokenizer.encode_plus(l, add_special_tokens=True, padding='max_length',
                                              max_length=max_length,
                                              return_attention_mask=True, return_tensors='pt')

            if encoded_i['input_ids'].size(dim=1) <= max_length and encoded_l['input_ids'].size(dim=1) <= max_length:
                in_ids.append(encoded_i['input_ids'])
                in_masks.append(encoded_i['attention_mask'])
                l_ids.append(encoded_l['input_ids'])
                l_masks.append(encoded_l['attention_mask'])
            else:
                filtered_count += 1

        # Print filtered
        print(f"Filtered {filtered_count} input-label-pairs as they exceeded max length {max_length}.")

        # Transform to tensors
        in_ids = torch.cat(in_ids, dim=0)
        in_masks = torch.cat(in_masks, dim=0)
        l_ids = torch.cat(l_ids, dim=0)
        l_masks = torch.cat(l_masks, dim=0)

        return in_ids, in_masks, l_ids, l_masks

    def forward_to_model(self, model, criterion, batch):
        in_ids, in_masks, l_ids, l_masks = batch
        outputs = model(input_ids=in_ids, attention_mask=in_masks,
                        decoder_input_ids=l_ids[:, :-1].contiguous(),
                        decoder_attention_mask=l_masks[:, :-1].contiguous(),
                        labels=l_ids[:, 1:].contiguous())

        loss = criterion(outputs.logits.reshape(-1, outputs.logits.shape[-1]), l_ids[:, 1:].reshape(-1))
        return outputs.logits, loss

    def eval(self, model, optimizer, tokenizer, criterion, data_set, metrics: TrainUtil.Metrics, split_into_two=False,
             log_prefix=None):
        super(TransductionTrainer, self).eval(model, optimizer, tokenizer, criterion, data_set, metrics, split_into_two,
                                              log_prefix)

        # Model specific evaluation


class LSTMTransductionMetrics(TrainUtil.Metrics):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.metrics = np.zeros(5)
        self.generator = model.generator
        self.errors = []

    def update(self, batch, outputs):

        # input_ids, in_masks, labels, _ = batch
        # in_masks = in_masks.cpu().detach()

        # Find nearest
        with torch.no_grad():
            predicted_indices = self.generator(outputs)
            predicted_indices = torch.argmax(predicted_indices, dim=-1)

        # Move to CPU
        input_ids, in_masks, labels, _ = (tensor.cpu().detach() for tensor in batch)
        predicted_indices = predicted_indices.cpu().detach()

        # Sample one erroneous prediction
        error_indices = torch.nonzero(torch.any(predicted_indices != labels, dim=1)).squeeze().tolist()
        if error_indices:
            if not isinstance(error_indices, list):
                error_indices = [error_indices]
            idx = error_indices[random.randrange(len(error_indices))]
            self.errors.append((input_ids[idx], labels[idx], predicted_indices[idx]))

        labels = labels.reshape(-1, 1)
        predicted_indices = predicted_indices.reshape(-1, 1)

        assert labels.size() == predicted_indices.size(), f"{labels.size()} != {predicted_indices.size()}"

        full_hit_perc = count_full_hit_percentage(labels=labels, predicted=predicted_indices.reshape(labels.size()))  # full hits



        #  Calculate
        avg_length = torch.count_nonzero(in_masks) / in_masks.size(0)
        acc = accuracy_score(y_true=labels, y_pred=predicted_indices)  # accuracy,
        f1 = f1_score(y_true=labels, y_pred=predicted_indices, average="micro")  # f1
        edit_dist = calculate_edit_distance(labels, predicted_indices)  # levenshtein

        # Update
        self.metrics += np.array([avg_length, acc, f1, edit_dist, full_hit_perc])

    def print(self, index, set_size, prefix, count, epoch, loss, decoder=None):
        avg_loss = loss / count
        metrics = self.metrics / count
        [avg_length, accuracy, f1, levenshtein, full_hit_perc] = metrics
        print(
            f"Batch {index}/{set_size} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Lengths: {avg_length:.2f}")

        error = self.errors[random.randrange(len(self.errors))]
        inp, label, pred = (decoder(ids) for ids in error)
        print(f"IN: \n\t{inp} \nOUT: \n\t{pred} \nTRUE: \n\t{label}\n")

        bracket_acc = 0.
        for error in self.errors:
            inp, label, pred = (decoder(ids) for ids in error)
            bracket_acc += bracket_accuracy(label, pred)

        bracket_acc /= len(self.errors)

        wandb.log({
            prefix + "in_lengths": avg_length,
            prefix + "epoch": epoch,
            prefix + "loss": avg_loss,
            prefix + "accuracy": accuracy,
            prefix + "f1": f1,
            prefix + "levenshtein": levenshtein,
            prefix + "full_hit%": full_hit_perc,
            prefix + "bracket_acc": bracket_acc
        })

    def reset(self):
        self.metrics = np.zeros(5)
        self.errors = []


class Seq2SeqLSTMTrainer(TransductionTrainer):
    def __init__(self, hp: Common.HyperParams, tokenizer: BartTokenizer):
        super(Seq2SeqLSTMTrainer, self).__init__(hp)
        self.tokenizer = tokenizer

    def decode_tokens(self, tokenizer: BartTokenizer, ids) -> str | list[str]:
        if len(ids.size()) == 1:
            return tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return [tokenizer.decode(ids[i], skip_special_tokens=True, clean_up_tokenization_spaces=True) for i in
                range(ids.size(0))]

    def encode(self, inputs, labels, tokenizer):
        self.tokenizer = tokenizer
        r = super().encode(inputs, labels, tokenizer)
        return r

    def forward_to_model(self, model, criterion, batch):
        in_ids, in_masks, l_ids, l_masks = batch
        pad_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        seq_length = in_ids.size(1)

        def num_pad_token(t):
            return torch.eq(t, pad_id).sum().item()

        def true_lengths(ids):
            return torch.tensor(
                [seq_length - num_pad_token(ids[b, :]) for b in range(ids.size(0))]
            )

        # Calculate Lengths
        in_len = true_lengths(in_ids)
        l_len = true_lengths(l_ids)

        # Create packed
        packed_in = pack_padded_sequence(in_ids, lengths=in_len, batch_first=True, enforce_sorted=False)
        packed_l = pack_padded_sequence(l_ids, lengths=l_len, batch_first=True, enforce_sorted=False)

        # Pass and return TODO: Pass packed_in & packed_l when Packed Sequence processing works
        out = model(in_ids, l_ids)
        prediction = model.generator(out)
        loss = criterion(prediction.contiguous().view(-1, prediction.size(-1)), l_ids.contiguous().view(-1))
        return out, loss


if __name__ == "__main__":
    print("Starting script...")
    start()
