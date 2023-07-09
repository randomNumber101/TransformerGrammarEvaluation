import json
import torch
from sklearn.preprocessing import LabelEncoder


def load_data(filepath):
    with open(filepath, 'r') as f:
        d = json.load(f)
    return d

