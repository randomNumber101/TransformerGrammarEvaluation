import json
import torch
from sklearn.preprocessing import LabelEncoder


def load_data(filepath):
    with open(filepath, 'r') as f:
        d = json.load(f)
    return d



def encode_for_tree_transduction(data, tokenizer):
    inputs = [entry['i'] for entry in data]
    labels = [entry['l'] for entry in data]

    input_ids = []
    label_ids = []
    attention_masks = []

    for input_text in inputs:
        encoded = tokenizer.encode_plus(input_text, add_special_tokens=True, max_length=128, pad_to_max_length=True,
                                        return_attention_mask=True, return_tensors='pt')
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_masks'])

    for label in labels:
        ()
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    # TODO: LabelEncoder must be external to reverse transformation


    return input_ids, attention_masks, encoded_labels

