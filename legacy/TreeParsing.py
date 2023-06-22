"""
    In:
        He goes to the mall.
    Out:
        ((He))((goes)((to)((the)(mall))))
"""
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

# Parse script arguments
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


def encode_for_tree_parsing(texts, tokenizer):
    ids = []
    masks = []

    filtered_count = 0

    max_length = hp.input_size if hp.input_size > 0 else tokenizer.model_max_length

    for text in texts:
        encoded = tokenizer.encode_plus(text, add_special_tokens=True, padding='max_length',
                                        max_length=max_length,
                                        # truncation=True, sequences > max_lengths will be omitted
                                        return_attention_mask=True, return_tensors='pt')
        if len(encoded['input_ids']) <= max_length:
            ids.append(encoded['input_ids'])
            masks.append(encoded['attention_mask'])
        else:
            filtered_count += 1

        ids.append(encoded['input_ids'])
        masks.append(encoded['attention_mask'])

    # Print filtered
    print(f"Filtered {filtered_count} input-label-pairs as they exceeded max length {max_length}.")

    # Transform to tensors
    ids = torch.cat(ids, dim=0)
    masks = torch.cat(masks, dim=0)
    return ids, masks


def split_and_prepare(tokenizer, dataset):
    # Sort by sequence length, log lengths
    data = dataset['data']
    data.sort(key=(lambda entry: len(entry.get('i'))))  # sort by input length
    set_size = len(data)

    # Log data metrics
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
    input_ids, input_masks = encode_for_tree_parsing(texts=inputs, tokenizer=tokenizer)
    label_ids, label_masks = encode_for_tree_parsing(texts=labels, tokenizer=tokenizer)

    # Split ids and masks for inputs and labels into test and training sets, respectively
    train = {}
    test = {}
    train['in_ids'], test['in_ids'], train['in_masks'], test['in_masks'], \
    train['l_ids'], test['l_ids'], train['l_masks'], test['l_masks'] = \
        train_test_split(input_ids, input_masks, label_ids, label_masks, test_size=0.2, random_state=420)

    train['set'] = TensorDataset(*itemgetter('in_ids', 'in_masks', 'l_ids', 'l_masks')(train))
    train['loader'] = DataLoader(train['set'], batch_size=hp.batch_size, shuffle=True, drop_last=True)

    test['set'] = TensorDataset(*itemgetter('in_ids', 'in_masks', 'l_ids', 'l_masks')(test))
    test['loader'] = DataLoader(test['set'], batch_size=hp.batch_size, shuffle=True, drop_last=True)
    return train, test


def calculate_metrics(labels, outputs):
    labels = labels.reshape(-1, 1)
    predicted_indices = torch.argmax(outputs, dim=-1).reshape(-1, 1)
    assert labels.size() == predicted_indices.size()

    return np.array([
        accuracy_score(y_true=labels, y_pred=predicted_indices),  # accuracy,
        f1_score(y_true=labels, y_pred=predicted_indices, average="micro"),  # f1
        calculate_edit_distance(labels, predicted_indices),
        count_full_hit_percentage(labels=labels, predicted=predicted_indices)
    ])


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

# Train & Evaluate Model
def train(model, optimizer, criterion, data_set, is_training=True, num_epochs=hp.num_epochs):
    device = hp.device
    for epoch in range(num_epochs):
        model.train() if is_training else model.eval()
        total_loss = 0
        metrics = np.zeros(4)

        print("Starting Epoch: {}/{}".format(epoch + 1, num_epochs))

        for i, batch in enumerate(data_set['loader']):
            in_ids, in_masks, l_ids, l_masks = batch

            # Move to device
            in_ids = in_ids.to(device=device)
            in_masks = in_masks.to(device=device)
            l_ids = l_ids.to(device=device)
            l_masks = l_masks.to(device=device)

            # forward
            optimizer.zero_grad()
            outputs = model(input_ids=in_ids, attention_mask=in_masks,
                            decoder_input_ids=l_ids[:, :-1].contiguous(),
                            decoder_attention_mask=l_masks[:, :-1].contiguous(),
                            labels=l_ids[:, 1:].contiguous())

            loss = criterion(outputs.logits.reshape(-1, outputs.logits.shape[-1]), l_ids[:, 1:].reshape(-1))

            total_loss += loss.item()
            metrics += calculate_metrics(labels=l_ids[:, 1:].cpu(), outputs=outputs.logits.cpu())

            if is_training:
                # back propagate
                loss.backward()

                # Apply gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=hp.max_norm)
                optimizer.step()

            # print info
            if (i + 1) % hp.print_every == 0 or (i + 1) == len(data_set["loader"]):
                # for last batch in epoch use residual, else use hp.print_every
                count = hp.print_every if (i + 1) % hp.print_every == 0 else len(data_set["loader"]) % hp.print_every
                avg_loss = total_loss / count
                metrics = metrics / count
                [accuracy, f1, levenshtein, full_hit_perc] = metrics
                print(
                    f"Batch {i + 1}/{len(data_set['loader'])} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

                prefix = "" if is_training else "eval_"
                wandb.log({
                    prefix + "epoch": epoch,
                    prefix + "loss": avg_loss,
                    prefix + "accuracy": accuracy,
                    prefix + "f1": f1,
                    prefix + "levenshtein": levenshtein,
                    prefix + "full_hit%": full_hit_perc
                })

                total_loss = 0
                metrics = np.zeros(4)
                if device == "cuda":
                    torch.cuda.empty_cache()


def save(name, model, optimizer):
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }, os.path.join(Common.result_folder, name + ".pth"))
    print("Saved model and optimizer to " + Common.result_folder + " with name: " + name + ".")


def main():
    # Testing
    l = torch.tensor([[3, 2, 3], [1, 2, 3]])
    i = torch.tensor([[1, 3, 3], [1, 2, 3]])
    print(calculate_edit_distance(l, i))

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
    train_set, test_set = split_and_prepare(tokenizer, dataset)

    # Define Model
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')  # BART Decoder-Encoder for Seq2Seq
    model.to(device=hp.device)

    # Initialize optimizer and criterion
    optimizer = AdamW(model.parameters(), lr=hp.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    criterion.to(hp.device)

    # Start training
    train(model=model, optimizer=optimizer, criterion=criterion, data_set=train_set)
    train(model=model, optimizer=optimizer, criterion=criterion, data_set=test_set, is_training=False, num_epochs=1)

    # Save Model & Optimizer
    save("BRACKETING_" + training_set_name.replace(".json", ""), model=model, optimizer=optimizer)


print("Starting script...")
if __name__ == "__main__":
    main()
