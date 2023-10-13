import argparse
import os
import re
import typing
from typing import ClassVar, Tuple

import numpy
import torch
from sklearn.preprocessing import LabelEncoder
from torch import Tensor
from transformers import PreTrainedTokenizer, BertTokenizer, BartTokenizer, BartTokenizerFast
import joblib

dirname = os.path.dirname(__file__)
training_folder = os.path.join(dirname, ".." + os.sep + "data" + os.sep)
tokenizer_folder = os.path.join(training_folder, "tokenizers")
tensor_data_folder = os.path.join(training_folder, "tensors")
result_folder = os.path.join(dirname, ".." + os.sep + ".." + os.sep + "server-import" + os.sep)


class HyperParams:
    def __init__(self):
        # default values
        self.device = "cpu"
        self.batch_size = 32
        self.input_size = -1
        self.num_epochs = 5

        self.learning_rate = 1e-4
        self.max_norm = -1  # Used for gradient clipping

        # Printing
        self.print_every = None


def overwrite_params(hps: HyperParams, overwrite: typing.Dict[str, object]):
    for (k, val) in overwrite.items():
        if val is None:
            continue
        if k == "device":
            hps.device = str(val)
        elif k == "batch_size":
            hps.batch_size = int(val)
        elif k == "input_size":
            hps.input_size = int(val)
        elif k == "num_epochs":
            hps.num_epochs = int(val)
        elif k == "learning_rate":
            hps.learning_rate = float(val)
        elif k == "max_norm":
            hps.max_norm = float(val)
        elif k == "print_every":
            hps.print_every = int(val)
    return hps


def loadParams():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--set_name', type=str, help="The data set", required=True)
    parser.add_argument('-m', '--model', type=str, help="Model to use [TRANSF, LSTM, BART]", default="TRANSF"),
    parser.add_argument('-bs', '--batch_size', type=int, help="Batch size")
    parser.add_argument('-is', '--input_size', type=int, help="Input (and output) Length of the network")
    parser.add_argument('-epochs', '--num_epochs', type=int, help="Epoch count")
    parser.add_argument('-lr', '--learning_rate', type=float, help="Learning rate")
    parser.add_argument('-mn', "--max_norm", type=float, help="Maximal gradient for gradient clipping")
    parser.add_argument('-pe', '--print_every', type=int, help="Frequency of logging results")
    parser.add_argument('-test', '--test_mode', help="Test mode. Deactivates wandb.", action='store_true')
    parser.add_argument('-l', '--layers', type=int, help="Number of layers", default=-1)
    parser.add_argument('-tok', '--tokenize', type=str, help="Tokenization strategy [bpe, words, pure_bpe]",
                        default="bpe")
    parser.add_argument('-eval', '--eval', help="Evaluation mode. Supply dataset.", action='store_true')
    parser.add_argument('--eval_from', help="Dataset to use for evaluation", type=str, default=None)
    parser.add_argument("-load", '--load', help="Loads models_data from pth file.", action='store_true')
    parser.add_argument('-ds', '--set_size', type=int, help="How much of the dataset to load. Default: -1.", default=-1)
    parser.add_argument('-bracket', '--bracket_depth', help="Sort eval set by bracket depth (labels).",
                        action='store_true')
    parser.add_argument('-pp', "--preprocess-only", help="Stops after pre-processing training data. Supply dataset.",
                        action='store_true')
    parser.add_argument('-project', '--wandb_project', type=str, help="Project name to upload data to.", default=None)

    parser.add_argument('-noopt', "--no_optimizer", help="Deactivate Noam Optimization (inverse sqrt lr scheduler).",
                        action='store_true')

    parser.add_argument('-nopre', "--no_pretrained_weight", help="Initialize models without pretrained weights", action="store_true")


    args = parser.parse_args()

    # Set script Hyper Parameters
    hp = HyperParams()
    overwrite_params(hp, vars(args))
    hp.device = "cuda" if torch.cuda.is_available() else "cpu"  # Check if CUDA is available

    print(f"hyper parameters: {vars(hp)}")
    return args, hp


def bracket_tokenizer_of(SupClass: type[PreTrainedTokenizer]):
    class BracketTokenizer(SupClass):
        def __init__(self, *args, **kwargs):
            kwargs["add_prefix_space"] = True
            super(BracketTokenizer, self).__init__(*args, **kwargs)
            # Regex: Either open bracket with a type (word), closed bracket or a string not containing brackets or space.

            self.token_pattern = r"\(\w*|\)|[^()]+"
            self.bracket_ids = {}
            self.next_bracket_id_index = 0
            self.bracket_types = set()
            self.do_word_wise = False
            self.grammar_vocab = set()

        def word_wise(self):
            self.encoder = {
                self.cls_token: 0,
                self.pad_token: 1,
                self.sep_token: 2,
                self.unk_token: 3,
                self.mask_token: 4
            }
            self.decoder = {
                0: self.cls_token,
                1: self.pad_token,
                2: self.sep_token,
                3: self.unk_token,
                4: self.mask_token
            }
            self.do_word_wise = True
            return self

        def get_bracket_token(self, bracket_type, opening):
            if not opening:
                t = ')' + bracket_type
            else:
                t = '(' + bracket_type

            if t not in self.bracket_types:
                self.bracket_types.add(t)
                self.add_tokens(t)

            return t

        def _add_token(self, token):
            if token not in self.grammar_vocab:
                self.grammar_vocab.add(token)
                self.add_tokens(token)

        def tokenize(self, text: str, return_depths=False, **kwargs):
            text = text.lower()
            basicTokens = re.findall(pattern=self.token_pattern, string=text)
            split_tokens = []
            bracket_stack = []
            for token in basicTokens:
                if token.startswith("("):
                    bracket_type = token[1:] if len(token) > 1 else "-NONE-"
                    bracket_stack.append(bracket_type)
                    token = self.get_bracket_token(bracket_type=bracket_type, opening=True)
                    self._add_token(token)
                    split_tokens.append(token)
                elif token == ')' and bracket_stack:
                    bracket_type = bracket_stack.pop()
                    token = self.get_bracket_token(bracket_type=bracket_type, opening=False)
                    self._add_token(token)
                    split_tokens.append(token)
                else:
                    if self.do_word_wise:
                        words = token.split()
                        for word in words:
                            self._add_token(word)
                            split_tokens.append(word)
                    else:
                        split_tokens.extend(super(BracketTokenizer, self)._tokenize(token))
            return split_tokens

        @classmethod
        def from_pretrained(cls, pretrained_model_name_or_path: typing.Union[str, os.PathLike], *args, **kwargs):
            return super(BracketTokenizer, cls).from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

    return BracketTokenizer


def save(path, name, model, optimizer=None):
    full_path = os.path.join(result_folder, path)
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer is not None else {}
    }, os.path.join(full_path, name + ".pth"))
    print("Saved model and optimizer to " + result_folder + " with name: " + name + ".")


def get_state_dict(file_name, path=None):
    path = path if path else "."
    path = os.path.join(result_folder, path, file_name + ".pth")
    if os.path.isfile(path):
        model_dict = torch.load(path)
        return model_dict['model']
    print(f"Could not load {file_name} from path {str(path)}")
    return None


def get_tokenizer_name(train_set: str, word_wise=False):
    return train_set.replace(".json", "") + "-" + ("words" if word_wise else "bpe")


def save_tokenizer(tokenizer: PreTrainedTokenizer, set_name, word_wise=False):
    tokenizer_name = get_tokenizer_name(set_name, word_wise)
    tokenizer.save_pretrained(os.path.join(tokenizer_folder, tokenizer_name))
    print(f"Successfully saved tokenizer for {tokenizer_name}.")


def load_tokenizer(baseclass: PreTrainedTokenizer, set_name, word_wise=False):
    path = os.path.join(tokenizer_folder, get_tokenizer_name(set_name, word_wise))
    if os.path.exists(path):
        tokenizer = baseclass.from_pretrained(path)
        if word_wise:
            return tokenizer.word_wise()
        return tokenizer
    return None


def save_label_encoder(encoder: LabelEncoder, tokenizer_name: str):
    path = os.path.join(tokenizer_folder, tokenizer_name)
    if not os.path.exists(path):
        os.makedirs(path)
    numpy.save(os.path.join(path, 'classes.npy'), encoder.classes_)


def load_label_encoder(tokenizer_name: str):
    path = os.path.join(tokenizer_folder, tokenizer_name, "classes.npy")
    if os.path.isfile(path):
        encoder = LabelEncoder()
        encoder.classes_ = numpy.load(path)
        return encoder
    return None


def save_tensor_data(tokenizer_name, data: Tuple[Tensor, Tensor, Tensor, Tensor], stats=None):
    in_ids, in_masks, l_ids, l_masks = data
    folder = os.path.join(tensor_data_folder, tokenizer_name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    torch.save((in_ids, in_masks), os.path.join(folder, "input.pt"))
    torch.save((l_ids, l_masks), os.path.join(tensor_data_folder, tokenizer_name, "label.pt"))
    print(f"Successfully saved preprocessed tensor data for {tokenizer_name}.")
    if stats:
        torch.save(stats,  os.path.join(tensor_data_folder, tokenizer_name, "stats.obj"))
        print(f"Successfully saved preprocessed statistics data for {tokenizer_name}.")

def load_tensor_data(tokenizer_name=None, return_stats=False):
    if not tokenizer_name:
        return None
    path = os.path.join(tensor_data_folder, tokenizer_name)
    if not os.path.exists(path):
        return None
    in_ids, in_masks = torch.load(os.path.join(path, "input.pt"))
    print(f"Successfully loaded pre-processed input data data from {tokenizer_name}.")
    l_ids, l_masks = torch.load(os.path.join(path, "label.pt"))
    print(f"Successfully loaded pre-processed label data data from {tokenizer_name}.")
    if not return_stats:
        return in_ids, in_masks, l_ids, l_masks
    try:
        stats = torch.load(os.path.join(tensor_data_folder, tokenizer_name, "stats.obj"))
    except Exception as err:
        print("No stats found in path. Preprocessing from json file instead.")
        return None
    print(f"Successfully loaded pre-processed statistics data data from {tokenizer_name}.")
    return (in_ids, in_masks, l_ids, l_masks), stats



def get_optimizer_dict(file):
    path = os.path.join(result_folder, file + ".pth")
    model_dict = torch.load(path)
    return model_dict['optimizer']
