# This is a sample Python script.
import json
import sys

import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, AdamW

from operator import itemgetter

from DataUtil import load_data
import Common

training_folder = Common.training_folder
training_set_name = "Grammar1_-7564333510142795128.json" if (len(sys.argv) == 1) else sys.argv[1]

# Hyper Parameters
hp = Common.HyperParams()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def encode_for_masking(data, tokenizer, mask_string, label_encoder):
    mask_token = tokenizer.mask_token
    '''
    In:
        He [MASK] to the mall.
    Out:
        goes     
    '''

    inputs = [entry['i'] for entry in data]
    labels = [entry['l'] for entry in data]

    input_ids = []
    attention_masks = []

    for input_text in inputs:
        input_text.replace(mask_string, mask_token)
        encoded = tokenizer.encode_plus(input_text, add_special_tokens=True, padding='max_length',
                                        return_attention_mask=True, return_tensors='pt')
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    # Encode labels
    encoded_labels = torch.tensor(label_encoder.fit_transform(labels))

    return input_ids, attention_masks, encoded_labels


# Load & Encode Training Data
dataset = load_data(training_folder + training_set_name)
encoder = LabelEncoder()
input_ids, attention_masks, encoded_labels = encode_for_masking(dataset['data'], tokenizer=tokenizer,
                                                                mask_string="[MASK]", label_encoder=encoder)

# Split into test and training
train = dict()
test = dict()
train['inputs'], test['inputs'], train['masks'], test['masks'], train['labels'], test['labels'] = \
    train_test_split(input_ids, attention_masks, encoded_labels, test_size=0.2, random_state=420)

train['set'] = TensorDataset(*itemgetter('inputs', 'masks', 'labels')(train))
train['loader'] = DataLoader(train['set'], batch_size=hp.batch_size, shuffle=True)

test['set'] = TensorDataset(*itemgetter('inputs', 'masks', 'labels')(test))
test['loader'] = DataLoader(train['set'], batch_size=hp.batch_size, shuffle=False)

# Define Model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')  # Bert + Linear Layer -> Classification
criterion = torch.nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=hp.learning_rate)

# Train & Evaluate Model
for epoch in range(hp.num_epochs):
    model.train()
    for batch in train['loader']:
        inputs, masks, labels = batch

        # Move to CUDA
        inputs = inputs.to(device="cuda")
        masks = masks.to(device="cuda")
        labels = labels.to(device="cuda")

        optimizer.zero_grad()
        outputs = model(inputs, attention_masks=masks, labels=labels)
        outputs.loss.backward()
        optimizer.step()

    # Evaluate
    model.eval()
    eval_loss = 0
    num_eval_steps = 0
    for batch in test['loader']:
        inputs, masks, labels = batch
        with torch.no_grad():
            outputs = model(inputs, attention_mask=masks, labels=labels)
            eval_loss += outputs.loss
            num_eval_steps += 1

    avg = eval_loss / num_eval_steps
    print('Epoch {}: Average Evaluation Loss: {}'.format(epoch + 1, avg))

model.save()
