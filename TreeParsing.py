"""
    In:
        He goes to the mall.
    Out:
        ((He))((goes)((to)((the)(mall))))
"""
import datetime
import random
import re

import textdistance
import torch
from sklearn.metrics import accuracy_score, f1_score

import TrainUtil
import wandb
import numpy as np

from transformers import BartForConditionalGeneration, BartTokenizer
from torch.optim import AdamW

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


class BracketTokenizer(BartTokenizer):
    def __init__(self, *args, **kwargs):
        super(BracketTokenizer, self).__init__(*args, **kwargs, add_prefix_space=True)
        # Regex: Either open bracket with a type (word), closed bracket or a string not containing brackets or space.
        self.token_pattern = r"\(\w*|\)|[^()\s]+"
        self.bracket_ids = {}
        self.next_bracket_id_index = 0

    # deprecated
    # returns even number for opening brackets and odd for closing. brackets of same type have succeeding numbers.
    def depr_get_bracket_id(self, bracket_type, opening):
        if bracket_type in self.bracket_ids:
            return self.bracket_ids.get(bracket_type) + int(not opening)
        else:
            newId = self.next_bracket_id_index
            self.next_bracket_id_index += 2
            return newId + int(not opening)

    def get_bracket_id(self, bracket_type, opening):
        if not opening:
            return ')' + bracket_type
        else:
            return '(' + bracket_type

    def _tokenize(self, text, **kwargs):
        basicTokens = re.findall(pattern=self.token_pattern, string=text)
        split_tokens = []
        bracket_stack = []
        for token in basicTokens:
            if token.startswith("("):
                bracket_type = token[1:] if len(token) > 1 else "-NONE-"
                bracket_stack.append(bracket_type)
                split_tokens.append(self.get_bracket_id(bracket_type=bracket_type, opening=True))
            elif token == ')':
                bracket_type = bracket_stack.pop()
                split_tokens.append(self.get_bracket_id(bracket_type=bracket_type, opening=False))
            else:
                split_tokens.extend(super()._tokenize(token))
        return split_tokens


def calculate_edit_distance(labels, predicted):
    index = random.randrange(labels.size(dim=0))
    return textdistance.levenshtein.distance(labels[index, :].tolist(), predicted[index, :].tolist()) * labels.size(
        dim=0) / labels.size(dim=1)


def calculate_edit_distance_all(labels, predicted):
    pairs = zip(labels.tolist(), predicted.tolist())
    return np.sum([textdistance.levenshtein.distance(entry[0], entry[1]) for entry in pairs])


def count_full_hit_percentage(labels, predicted):
    diff = torch.count_nonzero(labels - predicted, dim=1)
    non_full_hits = torch.count_nonzero(diff)
    num_full_hits = labels.size(0) - non_full_hits
    return num_full_hits.item() / labels.size(0)


class TransductionMetrics(TrainUtil.Metrics):

    def __init__(self):
        super().__init__()
        self.metrics = np.zeros(4)

    def update(self, batch, outputs):
        _, _, labels, _ = batch
        labels = labels[:, 1:].reshape(-1, 1).cpu().detach()
        outputs = outputs.logits.cpu().detach()
        predicted_indices = torch.argmax(outputs, dim=-1).reshape(-1, 1)

        assert labels.size() == predicted_indices.size(), f"{labels.size()} != {predicted_indices.size()}"

        #  Calculate
        acc = accuracy_score(y_true=labels, y_pred=predicted_indices)  # accuracy,
        f1 = f1_score(y_true=labels, y_pred=predicted_indices, average="micro")  # f1
        edit_dist = calculate_edit_distance(labels, predicted_indices)  # levenshtein
        full_hit_perc = count_full_hit_percentage(labels=labels, predicted=predicted_indices)  # full hits

        # Update
        self.metrics += np.array([acc, f1, edit_dist, full_hit_perc])

    def print(self, index, set_size, prefix, count, epoch, loss):
        avg_loss = loss / count
        metrics = self.metrics / count
        [accuracy, f1, levenshtein, full_hit_perc] = metrics
        print(
            f"Batch {index}/{set_size} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

        wandb.log({
            prefix + "epoch": epoch,
            prefix + "loss": avg_loss,
            prefix + "accuracy": accuracy,
            prefix + "f1": f1,
            prefix + "levenshtein": levenshtein,
            prefix + "full_hit%": full_hit_perc
        })

    def reset(self):
        self.metrics = np.zeros(4)


class TransductionTrainer(TrainUtil.Trainer):

    def __init__(self):
        super().__init__()

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

            encoded_l = tokenizer.encode_plus(i, add_special_tokens=True, padding='max_length',
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
        return outputs, loss


def main():
    # Initialize Trainer
    trainer = TransductionTrainer()
    hp = trainer.hp
    args = trainer.args

    # Load Training Data and Tokenizer
    training_set_name = "Grammar1_7143222753796263824.json" if not args.set_name else args.set_name
    tokenizer = BracketTokenizer.from_pretrained('facebook/bart-base')  # Use custom tokenizer
    dataset = load_data(Common.training_folder + training_set_name)  # Load data from file
    print("Loaded dataset: " + training_set_name)

    # Initialize weights & biases
    config = {
        "model": "BartForConditionalGeneration",
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
        "TreeBracketing",
        training_set_name.replace(".json", "")
    ]
    wandb.init(project="Tree Bracketing", config=config, tags=tags)
    wandb.define_metric("loss", summary='min')
    wandb.define_metric("accuracy", summary='max')

    # Split and prepare training data
    train_set, test_set = trainer.split_and_prepare(tokenizer, dataset['data'])

    # Define Model
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')  # BART Decoder-Encoder for Seq2Seq
    model.to(device=hp.device)

    # Initialize optimizer and criterion
    optimizer = AdamW(model.parameters(), lr=hp.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    criterion.to(hp.device)

    # Initialize Metrics
    metrics = TransductionMetrics()

    # Start training
    trainer.train(model=model, optimizer=optimizer, criterion=criterion, data_set=train_set, metrics=metrics)

    # Evaluate
    trainer.train(model=model, optimizer=optimizer, criterion=criterion, data_set=test_set, metrics=metrics,
                  is_training=False)

    # Save Model & Optimizer
    name = "BRACKETING_" + training_set_name.replace(".json", "") + "_" + str(datetime.datetime.now().timestamp())
    save(name, model=model, optimizer=optimizer)


if __name__ == "__main__":
    print("Starting script...")
    main()
