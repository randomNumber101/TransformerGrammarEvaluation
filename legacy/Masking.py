"""
    In:
        He goes to the mall.
    Out:
        ((He))((goes)((to)((the)(mall))))
"""
import argparse

import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

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


def encode_for_masking(inputs, labels, tokenizer, mask_string, label_encoder):
    mask_token = tokenizer.mask_token
    '''
    In:
        He [MASK] to the mall.
    Out:
        VP     
    '''
    input_ids = []
    attention_masks = []
    filtered_labels = []

    filtered_count = 0
    max_length = hp.input_size if hp.input_size > 0 else tokenizer.model_max_length

    for input_text, label in zip(inputs, labels):
        input_text.replace(mask_string, mask_token)

        encoded = tokenizer.encode_plus(input_text, add_special_tokens=True, padding='max_length',
                                        max_length=max_length,
                                        # truncation=True, sequences > max_lengths will be omitted
                                        return_attention_mask=True, return_tensors='pt')
        if len(encoded['input_ids']) <= max_length:
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
    encoded_labels = torch.tensor(label_encoder.fit_transform(filtered_labels))
    label_masks = torch.ones(encoded_labels.size())

    return input_ids, attention_masks, encoded_labels, label_masks


def split_and_prepare(tokenizer, dataset, label_encoder):
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
    inputs = [entry['i'] for entry in data]
    labels = [entry['l'] for entry in data]
    input_ids, attention_masks, encoded_labels, label_masks = encode_for_masking(inputs=inputs, labels=labels,
                                                                                 tokenizer=tokenizer,
                                                                                 mask_string="[MASK]",
                                                                                 label_encoder=label_encoder)

    # Split ids and masks for inputs and labels into test and training sets, respectively
    train = {}
    test = {}
    train['in_ids'], test['in_ids'], train['in_masks'], test['in_masks'], \
    train['l_ids'], test['l_ids'], train['l_masks'], test['l_masks'] = \
        train_test_split(input_ids, attention_masks, encoded_labels, label_masks, test_size=0.2, random_state=420)

    train['set'] = TensorDataset(*itemgetter('in_ids', 'in_masks', 'l_ids', 'l_masks')(train))
    train['loader'] = DataLoader(train['set'], batch_size=hp.batch_size, shuffle=True, drop_last=True)

    test['set'] = TensorDataset(*itemgetter('in_ids', 'in_masks', 'l_ids', 'l_masks')(test))
    test['loader'] = DataLoader(test['set'], batch_size=hp.batch_size, shuffle=True, drop_last=True)
    return train, test


def count_full_hit_percentage(labels, predicted):
    non_full_hits = torch.count_nonzero(labels - predicted)
    num_full_hits = labels.size(0) - non_full_hits
    return num_full_hits.item() / labels.size(0)


def calculate_metrics(labels, outputs):
    predicted = torch.argmax(outputs, dim=1)
    acc = accuracy_score(y_true=labels, y_pred=predicted) # accuracy,
    f1 = f1_score(y_true=labels, y_pred=predicted, average="micro")
    full_hit_perc = count_full_hit_percentage(labels, predicted)  # hit %
    return np.array([acc, f1, full_hit_perc])


# Train & Evaluate Model
def train(model, optimizer, criterion, data_set, is_training=True, num_epochs=hp.num_epochs):
    device = hp.device
    for epoch in range(num_epochs):
        model.train() if is_training else model.eval()
        total_loss = 0
        metrics = np.zeros(3)

        print("Starting Epoch: {}/{}".format(epoch + 1, num_epochs))

        for i, batch in enumerate(data_set['loader']):
            in_ids, in_masks, l_ids, l_masks = batch

            # Move to device
            in_ids = in_ids.to(device=device)
            in_masks = in_masks.to(device=device)
            l_ids = l_ids.to(device=device)
            l_masks = l_masks.to(device=device)  # not needed for this task

            # forward
            optimizer.zero_grad()
            outputs = model(in_ids, attention_mask=in_masks, labels=l_ids)
            logits = outputs.logits
            loss = criterion(logits, l_ids)

            total_loss += loss.item()
            metrics += calculate_metrics(labels=l_ids.cpu().detach(), outputs=logits.cpu().detach())

            if is_training:
                # back propagate
                loss.backward()
                # Apply gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=hp.max_norm)
                # update optimizer
                optimizer.step()

            # print info
            if (i + 1) % hp.print_every == 0 or (i + 1) == len(data_set["loader"]):
                # for last batch in epoch use residual, else use hp.print_every
                count = hp.print_every if (i + 1) % hp.print_every == 0 else len(data_set["loader"]) % hp.print_every

                #TODO: Merge using "printMetrics(count, loss, metrics)"
                avg_loss = total_loss / count
                metrics = metrics / count
                [accuracy, f1, full_hit_perc] = metrics
                print(
                    f"Batch {i + 1}/{len(data_set['loader'])} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

                prefix = "" if is_training else "eval_"
                wandb.log({
                    prefix + "epoch": epoch,
                    prefix + "loss": avg_loss,
                    prefix + "accuracy": accuracy,
                    prefix + "f1": f1,
                    prefix + "full_hit%": full_hit_perc
                })

                total_loss = 0
                metrics = np.zeros(3)
                if device == "cuda":
                    torch.cuda.empty_cache()


def main():
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
    label_encoder = LabelEncoder()
    train_set, test_set = split_and_prepare(tokenizer, dataset, label_encoder)

    # Define Model
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                          num_labels=len(label_encoder.classes_))  # BERT + Linear Layer
    model.to(device=hp.device)

    # Initialize optimizer and criterion
    optimizer = AdamW(model.parameters(), lr=hp.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    criterion.to(hp.device)

    # Start training & evaluation
    train(model=model, optimizer=optimizer, criterion=criterion, data_set=train_set)
    train(model=model, optimizer=optimizer, criterion=criterion, data_set=test_set, is_training=False, num_epochs=1)

    # Save Model & Optimizer
    save("MASKING_" + training_set_name.replace(".json", ""), model=model, optimizer=optimizer)


if __name__ == "__main__":
    print("Starting script...")
    main()
