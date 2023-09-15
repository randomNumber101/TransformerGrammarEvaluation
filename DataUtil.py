import json
import os.path

import fairseq.tasks.translation
import torch
from sklearn.preprocessing import LabelEncoder
from fairseq.data import FairseqDataset, LanguagePairDataset
from fairseq.tasks import translation
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from transformers import BartTokenizer

import Common


def load_data(filepath):
    with open(filepath, 'r') as f:
        d = json.load(f)
    return d


def convert_to_single_files(dest_dir, dataset_name, data_dict, tokenizer, test_portion=0.3, valid_portion=0.1):
    data = data_dict["data"]
    dest_dir = os.path.join(dest_dir, dataset_name)

    train_valid, test = train_test_split(data, shuffle=True, test_size=test_portion)
    train, valid = train_test_split(train_valid, test_size=valid_portion)

    is_classify_task = not ("bracket" in dataset_name.lower())

    save_to_single_files(dest_dir, "train", train, tokenizer=tokenizer, classify=is_classify_task)
    save_to_single_files(dest_dir, "test", test, tokenizer=tokenizer, classify=is_classify_task)
    save_to_single_files(dest_dir, "valid", valid, tokenizer=tokenizer, classify=is_classify_task)


def save_to_single_files(dest_dir, split_name, entries, tokenizer=None, max_tokens=512, classify=False):
    input_file_name = f"{split_name}.input"
    label_file_name = f"{split_name}.label"

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    input_file_path = os.path.join(dest_dir, input_file_name)
    label_file_path = os.path.join(dest_dir, label_file_name)

    def transform(x: str):
        if tokenizer:
            tokenized = tokenizer.tokenize(x)
            if len(tokenized) >= max_tokens:
                return None
            return " ".join(tokenized)
        return x

    with open(input_file_path, 'w', encoding='utf-8') as input_file, open(label_file_path, 'w', encoding='utf-8') as label_file:
        count = len(entries)
        skipped = 0
        for i, entry in enumerate(entries):

            t_in = transform(entry["i"])
            l_in = transform(entry["l"]) if not classify else entry["l"]

            if t_in and l_in:
                input_file.write(t_in + "\n")
                label_file.write(l_in + "\n")
            else:
                skipped += 1
            if i % 1000 == 0:
                print(f"Done with {i}/{count}({i * 100/count}%) - Skipped {skipped} entries.")


def tokenize_set(path, name: str, word_wise=False):
    tokenizer = Common.bracket_tokenizer_of(BartTokenizer).from_pretrained("facebook/bart-base")
    if word_wise:
        tokenizer = tokenizer.word_wise()
    output_dir_name = "fairseq-sets" + os.sep + ("bpe" if not word_wise else "words")
    output_dir = os.path.join(path, output_dir_name)
    file = os.path.join(path, name)
    name = name.replace(".json", "")
    convert_to_single_files(output_dir, name, load_data(file), tokenizer)


def tokenize_all():
    data_dir = Common.training_folder
    files = os.listdir(data_dir)
    for file in files:
        if not file.endswith(".json"):
            continue
        tokenize_set(data_dir, file, word_wise=False)
        tokenize_set(data_dir, file, word_wise=True)
        print("Tokenized: " + file)


# x = fairseq.data.Dictionary()
# x.encode_line()
class FairseqWrapper(FairseqDataset):
    def __init__(self, data: TensorDataset):
        super(FairseqWrapper, self).__init__()
        self.data = data

    def collater(self, samples):
        columns = len(samples[0])
        return (torch.stack(tensors, dim=0) for tensors in
                [[sample[column] for sample in samples] for column in range(columns)])

    def num_tokens(self, index):
        return self.data[index][0][1]

    def num_tokens_vec(self, indices):
        return NotImplementedError("Num_tokens is not implemented.")

    def size(self, index):
        return (len(tensor) for tensor in self.data[index])

    def prefetch(self, indices):
        pass

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    @property
    def supports_prefetch(self):
        return False


if __name__ == "__main__":
    print("Starting script...")
    build_preprocessed()
    print("Done")