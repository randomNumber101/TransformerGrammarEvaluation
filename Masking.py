"""
    In:
        He goes to the mall.
    Out:
        ((He))((goes)((to)((the)(mall))))
"""
import argparse
import datetime
from datetime import date

import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

import TrainUtil
import wandb
import numpy as np

from sklearn.model_selection import train_test_split

from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
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


def count_full_hit_percentage(labels, predicted):
    non_full_hits = torch.count_nonzero(labels - predicted)
    num_full_hits = labels.size(0) - non_full_hits
    return num_full_hits.item() / labels.size(0)


class ClassificationMetrics(TrainUtil.Metrics):
    def __init__(self):
        super().__init__()
        self.metrics = np.zeros(4)

    def update(self, batch, outputs):
        _, in_masks, labels, _ = batch

        # Move to CPU
        in_masks = in_masks.cpu().detach()
        labels = labels.cpu().detach()
        outputs = outputs.logits.cpu().detach()

        #  Calculate
        avg_lengths = torch.count_nonzero(in_masks) / in_masks.size(0)
        predicted = torch.argmax(outputs, dim=1)
        acc = accuracy_score(y_true=labels, y_pred=predicted)  # accuracy,
        f1 = f1_score(y_true=labels, y_pred=predicted, average="micro")
        full_hit_perc = count_full_hit_percentage(labels, predicted)  # hit %

        # Update
        self.metrics += np.array([avg_lengths, acc, f1, full_hit_perc])

    def print(self, index, set_size, prefix, count, epoch, loss):
        avg_loss = loss / count
        metrics = self.metrics / count
        [avg_lengths, accuracy, f1, full_hit_perc] = metrics
        print(
            f"Batch {index}/{set_size} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Lengths: {avg_lengths:.2f}")

        wandb.log({
            prefix + "in_lengths": avg_lengths,
            prefix + "epoch": epoch,
            prefix + "loss": avg_loss,
            prefix + "accuracy": accuracy,
            prefix + "f1": f1,
            prefix + "full_hit%": full_hit_perc
        })

    def reset(self):
        self.metrics = np.zeros(4)


class ClassificationTrainer(TrainUtil.Trainer):

    def __init__(self, hp, label_encoder, mask_string="[MASK]"):
        super().__init__(hp)
        self.label_encoder = label_encoder
        self.mask_string = mask_string

    def encode(self, inputs, labels, tokenizer):
        mask_token = tokenizer.mask_token

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


def main():
    args, hp = Common.loadParams()

    # Initialize Trainer
    label_encoder = LabelEncoder()
    trainer = ClassificationTrainer(hp, label_encoder=label_encoder)

    # Load Training Data and Tokenizer
    training_set_name = "Grammar1_7143222753796263824.json" if not args.set_name else args.set_name
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = load_data(Common.training_folder + training_set_name)  # Load data from file
    print("Loaded dataset: " + training_set_name)

    # Initialize weights & biases
    config = {
        "model": "BertForSequenceClassification",
        "optimizer": "AdamW",
        "criterion": "CrossEntropy",
        "training_set": training_set_name,

        "batch_size": hp.batch_size,
        "input_size": hp.input_size,
        "batch_eval_count": hp.print_every,

        "num_epochs": hp.num_epochs,
        "learning_rate": hp.learning_rate,
        "gradient_clipping_max": hp.max_norm
    }
    tags = [
        "Masking",
        training_set_name.replace(".json", "")
    ]
    wandb.init(project="Masking", config=config, tags=tags)
    wandb.define_metric("loss", summary='min')
    wandb.define_metric("accuracy", summary='max')

    # Split and prepare training data
    train_set, test_set = trainer.split_and_prepare(tokenizer, dataset['data'])

    # Define Model
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                          num_labels=len(label_encoder.classes_))  # BERT + Linear Layer
    model.to(device=hp.device)

    # Initialize optimizer and criterion
    optimizer = AdamW(model.parameters(), lr=hp.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    criterion.to(hp.device)

    # Initialize metrics
    metrics = ClassificationMetrics()

    # Start training
    trainer.train(model=model, optimizer=optimizer, criterion=criterion, data_set=train_set, metrics=metrics)

    # Eval
    trainer.eval(model=model, optimizer=optimizer, criterion=criterion, data_set=test_set, metrics=metrics,
                 split_into_two=True)

    # Save Model & Optimizer
    name = "MASKING_" + training_set_name.replace(".json", "") + "_" + str(datetime.datetime.now().timestamp())
    save(name, model=model, optimizer=optimizer)


if __name__ == "__main__":
    print("Starting script...")
    main()
