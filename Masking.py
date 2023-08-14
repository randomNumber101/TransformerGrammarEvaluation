"""
    In:
        He goes to the mall.
    Out:
        ((He))((goes)((to)((the)(mall))))
"""
import argparse
import datetime
import random
import re
from datetime import date
from typing import List

import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

import BaseLines
import ModelImpl
import TrainUtil
import wandb
import numpy as np

from sklearn.model_selection import train_test_split

from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, BartTokenizer, BertConfig, BertTokenizerFast
from torch.optim import AdamW

from operator import itemgetter

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


def main():
    args, hp = Common.loadParams()
    model_name = "BERT" if args.model == "TRANSF" else args.model
    model_postfix = "SMALL" if args.layers > 0 else "PRETRAINED"

    # Load Training Data and Tokenizer
    training_set_name = "Grammar1_7143222753796263824.json" if not args.set_name else args.set_name
    dataset = load_data(Common.training_folder + training_set_name)  # Load data from file
    print("Loaded dataset: " + training_set_name)

    # Initialize Tokenizer
    if args.tokenize == "words":
        BracketTokenizer = Common.bracket_tokenizer_of(BertTokenizer)
        tokenizer = BracketTokenizer.from_pretrained('bert-base-uncased').word_wise()
    elif args.tokenize == "words_bpe":
        BracketTokenizer = Common.bracket_tokenizer_of(BertTokenizer)
        tokenizer = BracketTokenizer.from_pretrained('bert-base-uncased')  # Use custom tokenizer
    else:
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')  # Use default tokenizer

    # Initialize Trainer
    label_encoder = LabelEncoder()
    if model_name == "BERT":
        trainer = ClassificationTrainer(hp, label_encoder=label_encoder)
    elif model_name == "LSTM":
        trainer = LSTMClassificationTrainer(hp, tokenizer, label_encoder)
    elif model_name == "SIMPLE":
        trainer = SimpleClassificationTrainer(hp, label_encoder=label_encoder)
    else:
        print("No such model: " + model_name)
        return

    # Initialize weights & biases
    config = {
        "model": model_name,
        "optimizer": "AdamW",
        "criterion": "CrossEntropy",
        "training_set": training_set_name,

        "batch_size": hp.batch_size,
        "input_size": hp.input_size,
        "batch_eval_count": hp.print_every,

        "num_epochs": hp.num_epochs,
        "learning_rate": hp.learning_rate,
        "gradient_clipping_max": hp.max_norm,

        "tokenization": args.tokenize,
        "layers": args.layers
    }
    tags = [
        "Classify",
        training_set_name.replace(".json", ""),
        model_name,
        model_postfix,
        args.tokenize
    ]
    wandb_mode = 'disabled' if args.test_mode else 'online'
    wandb.init(project="Classify", config=config, tags=tags, mode=wandb_mode)
    wandb.define_metric("loss", summary='min')
    wandb.define_metric("accuracy", summary='max')

    # Split and prepare training data
    train_set, test_set = trainer.split_and_prepare(tokenizer, dataset['data'], cap_size=args.set_size)

    # Define Model
    if model_name == "BERT":
        config = BertConfig.from_pretrained('bert-base-uncased', )
        if args.layers <= 0:
            args.layers = config.num_hidden_layers
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                              num_labels=len(label_encoder.classes_),
                                                              num_hidden_layers=args.layers
                                                              )  # BERT + Linear Layer
        model.resize_token_embeddings(len(tokenizer))
        metrics = ClassificationMetrics(label_encoder)

    elif model_name == "SIMPLE":
        layers = 3 if args.layers < 0 else args.layers
        model = BaseLines.SimpleClassifier(len(tokenizer), len(label_encoder.classes_),
                                           hidden_dim=512,
                                           num_layers=layers)
        metrics = ClassificationMetrics(label_encoder)

    elif model_name == "LSTM":
        model = BaseLines.BiLSTMClassifier(len(tokenizer), len(label_encoder.classes_), input_dim=256, hidden_dim=1024)
        metrics = ClassificationMetrics(label_encoder)

    else:
        raise NotImplementedError("No such model implemented: " + model_name)
    model.to(device=hp.device)

    # Initialize optimizer and criterion
    optimizer = AdamW(model.parameters(), lr=hp.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    criterion.to(hp.device)

    # Start training
    trainer.train(model=model, optimizer=optimizer, tokenizer=tokenizer, criterion=criterion, data_set=train_set,
                  metrics=metrics)

    # Eval
    trainer.test(model=model, optimizer=optimizer, tokenizer=tokenizer, criterion=criterion, data_set=test_set,
                 metrics=metrics,
                 split_into_two=False)

    # Save Model & Optimizer
    name = "MASKING_" + \
           training_set_name.replace(".json", "") + \
           model_name + "_" + \
           model_postfix + \
           args.tokenize

    save(name, model=model, tokenizer=tokenizer)


def count_full_hit_percentage(labels, predicted):
    non_full_hits = torch.count_nonzero(labels - predicted)
    num_full_hits = labels.size(0) - non_full_hits
    return num_full_hits.item() / labels.size(0)


def certainty(y_true, y_pred):
    count = y_true.size(0)
    y_pred = torch.softmax(y_pred, dim=-1)
    certainties = [y_pred[i, y_true[i].item()] for i in range(count)]
    avg_certainty = float(sum(certainties)) / count
    return avg_certainty


class ClassificationMetrics(TrainUtil.Metrics):
    def __init__(self, label_encoder: LabelEncoder):
        super().__init__()
        self.metrics = np.zeros(6)
        self.log_num = 0
        self.current_prefix = "<init>"
        self.decode = lambda x: label_encoder.inverse_transform(x)
        self.last_prefix = ""
        self.current_step = 0

    def update(self, batch, outputs):
        input_ids, in_masks, labels, _ = batch

        # Move to CPU
        in_masks = in_masks.cpu().detach()
        labels = labels.cpu().detach()
        outputs = outputs.cpu().detach()
        predicted = torch.argmax(outputs, dim=1)

        # Sample one erroneous prediction
        error_indices = torch.nonzero(predicted - labels).squeeze().tolist()
        error_indices = [error_indices] if not isinstance(error_indices, list) else error_indices
        err_cert = 0
        if error_indices and len(error_indices) > 0:
            idx = error_indices[random.randrange(len(error_indices))]
            self.errors.append((input_ids[idx], labels[idx], predicted[idx]))
            err_cert = certainty(y_true=predicted[error_indices], y_pred=outputs[error_indices])

        #  Calculate
        avg_lengths = torch.count_nonzero(in_masks) / in_masks.size(0)
        cert = certainty(y_true=labels, y_pred=outputs)
        acc = accuracy_score(y_true=labels, y_pred=predicted)  # accuracy,
        f1 = f1_score(y_true=labels, y_pred=predicted, average="micro")
        full_hit_perc = count_full_hit_percentage(labels, predicted)  # hit %

        # Update
        self.metrics += np.array([avg_lengths, acc, f1, full_hit_perc, cert, err_cert])

    def print(self, index, set_size, prefix, count, epoch, loss, decoder=None):

        if prefix != self.last_prefix:
            self.last_prefix = prefix
            self.current_step = 0

        avg_loss = loss / count
        metrics = self.metrics / count
        [avg_lengths, accuracy, f1, full_hit_perc, cert, err_cert] = metrics
        print(
            f"Batch {index}/{set_size} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Lengths: {avg_lengths:.2f}")

        if len(self.errors) > 0:
            def decode(er):
                i, l, p = er
                i = decoder(i)
                l, p = self.decode(l.unsqueeze(0)), self.decode(p.unsqueeze(0))
                return i, l, p

            error = self.errors[random.randrange(len(self.errors))]
            inp, label, pred = decode(error)
            print(f"IN: \n\t{inp} \nOUT: \n\t{pred} \nTRUE: \n\t{label}\n")

            for error in self.errors:
                inp, label, pred = decode(error)
                # TODO: Task specific metrix

        wandb.log({
            prefix + "in_lengths": avg_lengths,
            prefix + "epoch": epoch,
            prefix + "loss": avg_loss,
            prefix + "accuracy": accuracy,
            prefix + "f1": f1,
            prefix + "full_hit%": full_hit_perc,
            prefix + "certainty": cert,
            prefix + "stubbornness": err_cert,
            prefix + "custom_step": self.current_step
        })

        self.current_step += 1

    def reset(self):
        self.metrics = np.zeros(6)
        self.errors = []


class ClassificationTrainer(TrainUtil.Trainer):

    def generate(self, model, batch):
        print("Warning: The Generation method doesn't do much for Classification Tasks.")
        return model(batch)

    def __init__(self, hp, label_encoder, mask_string="[MASK]"):
        super().__init__(hp)
        self.label_encoder = label_encoder
        self.mask_string = mask_string

    def update_metrics(self, metrics: TrainUtil.Metrics, batch, outputs):
        metrics.update(batch, outputs.logits)

    def decode_single(self, tokenizer: BertTokenizer, x):
        if x is None or len(x.size()) < 1:
            return ""

        special_tks = tokenizer.all_special_tokens
        special_tks.remove(tokenizer.mask_token)

        with_special_tks = tokenizer.decode(x, skip_special_tokens=False, clean_up_tokenization_spaces=True)
        for spc_tkn in special_tks:
            with_special_tks = with_special_tks.replace(spc_tkn, "")
        return with_special_tks

    def decode_tokens(self, tokenizer: BertTokenizer, ids) -> str | list[str]:

        if len(ids.size()) == 1:
            return self.decode_single(tokenizer, ids)
        return [self.decode_single(tokenizer, ids[i]) for i in
                range(ids.size(0))]

    def encode(self, inputs, labels, tokenizer):
        mask_token = tokenizer.mask_token
        mask_id = tokenizer.mask_token_id

        input_ids = []
        attention_masks = []
        filtered_labels = []

        filtered_count = 0
        max_length = self.hp.input_size if self.hp.input_size > 0 else tokenizer.model_max_length

        for input_text, label in zip(inputs, labels):
            input_text.replace(self.mask_string, mask_token)

            encoded = tokenizer.encode_plus(input_text, add_special_tokens=True, padding='max_length',
                                            max_length=max_length,
                                            # truncation=True, sequences > max_lengths will be omitted
                                            return_attention_mask=True, return_tensors='pt')
            if encoded['input_ids'].size(dim=1) <= max_length:
                input_ids.append(encoded['input_ids'])
                attention_masks.append(encoded['attention_mask'])
                filtered_labels.append(label)
            else:
                filtered_count += 1

        # Print filtered
        print(f"Filtered {filtered_count} input-label-pairs as they exceeded max length {max_length}.")

        # Transform to tensors
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)

        # Encode labels
        encoded_labels = torch.tensor(self.label_encoder.fit_transform(filtered_labels))
        label_masks = torch.ones(encoded_labels.size())

        return input_ids, attention_masks, encoded_labels, label_masks

    def forward_to_model(self, model, criterion, batch):
        in_ids, in_masks, enc_labels, _ = batch
        outputs = model(in_ids, attention_mask=in_masks, labels=enc_labels)
        loss = criterion(outputs.logits, enc_labels)
        return outputs, loss


class SimpleClassificationTrainer(ClassificationTrainer):

    def update_metrics(self, metrics: TrainUtil.Metrics, batch, outputs):
        metrics.update(batch, outputs)
    def forward_to_model(self, model: BaseLines.SimpleClassifier, criterion, batch):
        in_ids, in_masks, enc_labels, _ = batch
        outputs = model.forward(in_ids, in_masks)
        loss = criterion(outputs, enc_labels)
        return outputs, loss


class LSTMClassificationTrainer(ClassificationTrainer):

    def update_metrics(self, metrics: TrainUtil.Metrics, batch, outputs):
        metrics.update(batch, outputs)

    def __init__(self, hp, tokenizer, label_encoder):
        super().__init__(hp, label_encoder)
        self.tokenizer = tokenizer

    def forward_to_model(self, model: BaseLines.BiLSTMClassifier, criterion, batch):
        in_ids, in_masks, labels, l_masks = batch
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

        # Pass and return
        out = model(in_ids, in_masks, in_len)
        loss = criterion(out.contiguous().view(-1, out.size(-1)), labels.contiguous().view(-1))
        return out, loss


if __name__ == "__main__":
    print("Starting script...")
    main()
