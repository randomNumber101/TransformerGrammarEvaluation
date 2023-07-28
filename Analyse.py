import torch
from transformers import PreTrainedModel

import Common


def load_model(model: PreTrainedModel, path):
    model.load_state_dict(Common.load_state_dict(path))
