"""
    In:
        He goes to the mall.
    Out:
        ((He))((goes)((to)((the)(mall))))
"""
import copy
import os
import random
import re

import textdistance
import torch
import torchmetrics.functional
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.nn import CrossEntropyLoss

import BaseLines
import ModelImpl
import TrainUtil
import wandb
import numpy as np

from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig, PreTrainedTokenizer, \
    BartTokenizerFast, BartModel
from torch.optim import AdamW

from ModelImpl import TokenEmbedder
from Common import save
from DataUtil import load_data
import Common

EPS = 0.05  ## Error EPS
bracket_pattern = r"\(\w*|\)\w*"


def deleteBartLayers(model: BartForConditionalGeneration, num_layers_to_keep):  # must pass in the full bert model
    oldEncoderList = model.model.encoder.layers
    oldDecoderList = model.model.decoder.layers
    newEncoderList = nn.ModuleList()
    newDecoderList = nn.ModuleList()

    # Now iterate over all layers, only keeping the relevant layers.
    for i in range(0, num_layers_to_keep):
        newEncoderList.append(oldEncoderList[i])
        newDecoderList.append(oldDecoderList[i])

    # create a copy of the model, modify it with the new list, and return
    copyOfModel = copy.deepcopy(model)
    copyOfModel.model.encoder.layers = newEncoderList
    copyOfModel.model.decoder.layers = newEncoderList
    return copyOfModel


def start():
    args, hp = Common.loadParams()
    model_name = "BART" if args.model == "TRANSF" else args.model
    model_postfix = "SMALL" if args.layers > 0 else "PRETRAINED"

    # Load Training Data
    training_set_name = "Grammar1_7143222753796263824.json" if not args.set_name else args.set_name
    tokenizer_set_name = training_set_name if not args.tokenizer_name else args.tokenizer_name
    dataset = load_data(Common.training_folder + training_set_name)  # Load data from file
    print("Loaded dataset: " + training_set_name)

    # Initialize Tokenizer
    save_tokenizer = False
    word_wise = False
    if args.tokenize == "words":
        TokenizerClass = Common.bracket_tokenizer_of(BartTokenizer)
        word_wise = True
    elif args.tokenize == "pure_bpe":
        TokenizerClass = BartTokenizerFast
    else:
        TokenizerClass = Common.bracket_tokenizer_of(BartTokenizer)

    tokenizer = Common.load_tokenizer(TokenizerClass, tokenizer_set_name, word_wise=word_wise)
    current_tokenizer_name = Common.get_tokenizer_name(tokenizer_set_name, word_wise)
    tokenizer_name = Common.get_tokenizer_name(training_set_name, word_wise)
    pretrained_name = "facebook/bart-base" if not args.tok_lang == "de" else "Shahm/bart-german"

    if not tokenizer:
        tokenizer = TokenizerClass.from_pretrained(pretrained_name)  # Use default tokenizer
        if word_wise:
            tokenizer = tokenizer.word_wise()
        save_tokenizer = True
        print(
            f"No saved tokenizer with name {current_tokenizer_name} has been found in local directory {Common.tokenizer_folder}. A new tokenizer will be trained.")
    else:
        print(f"Successfully loaded tokenizer {current_tokenizer_name} from local files.")

    if model_name == "BART":
        trainer = TransductionTrainer(hp)
    elif model_name == "SIMPLE" or model_name == "SIMPLE_BI":
        trainer = SimpleTransductionTrainer(hp)
    elif model_name == "LSTM":
        trainer = Seq2SeqLSTMTrainer(hp, tokenizer)
    else:
        print("No such model: " + model_name)
        return

    # Initialize weights & biases for logging
    config = {
        "model": (
                     "BartForConditionalGeneration" if model_name == "BART" else model_name
                 ) +
                 (
                     "" if not args.no_pretrained_weight else "-SCRATCH"
                 ),
        "optimizer": "AdamW",
        "criterion": "CrossEntropy",
        "training_set": training_set_name,
        "training_set_size": len(dataset['data']),

        "batch_size": hp.batch_size,
        "input_size": hp.input_size,
        "batch_eval_count": hp.print_every,
        "tokenization": args.tokenize,

        "num_epochs": hp.num_epochs,
        "learning_rate": hp.learning_rate,
        "gradient_clipping_max": hp.max_norm,
        "num_layers": args.layers
    }
    tags = [
        "TreeBracketing",
        training_set_name.replace(".json", ""),
        model_name + ("" if not args.no_pretrained_weight else "-SCRATCH"),
        model_postfix,
        args.tokenize
    ]
    if args.eval_from:
        tags.append(args.eval_from)
    wandb_mode = 'disabled' if args.test_mode else 'online'
    wandb.init(
        project="Tree Bracketing" if not args.wandb_project else args.wandb_project,
        config=config, tags=tags,
        mode=wandb_mode
    )
    wandb.define_metric("loss", summary='min')
    wandb.define_metric("accuracy", summary='max')

    # Split and prepare training data
    test_portion = 0.3
    if args.eval_from:
        evaluation_set_name = args.eval_from
        dataset = load_data(Common.training_folder + evaluation_set_name)  # Load data from file
        print("Loaded evaluation dataset: " + evaluation_set_name)
        tokenizer_name = Common.get_tokenizer_name(evaluation_set_name, word_wise)
        test_portion = 0.99
    force_encoding = args.tokenizer_name is not None
    train_set, test_set = trainer.split_and_prepare(tokenizer, dataset['data'], tokenizer_name, force_encoding=force_encoding,
                                                    cap_size=args.set_size, test_portion=test_portion)
    if save_tokenizer and not args.tokenize == "words_bpe":
        Common.save_tokenizer(tokenizer, training_set_name, word_wise)

    print("Preprocessing done.")
    if args.preprocess_only:
        return



    # Define Model
    if model_name == "BART":
        config = BartConfig.from_pretrained(pretrained_name)

        if not args.no_pretrained_weight:
            model = BartForConditionalGeneration.from_pretrained(
                pretrained_name)  # BART Decoder-Encoder for Seq2Seq
        else:
            # Load model without pretrained weights
            config = BartConfig()
            model = BartForConditionalGeneration(config)
        if args.layers <= 0:
            args.layers = config.num_hidden_layers
        else:
            model = deleteBartLayers(model, args.layers)
        model.resize_token_embeddings(len(tokenizer))
        metrics = BasicTransductionMetrics()
        optimizer = AdamW(model.parameters(), lr=hp.learning_rate, betas=(0.9, 0.98))
        if not args.no_optimizer:
            optimizer = TrainUtil.NoamOptim(optimizer, config.d_model)  # Use sqrt lr scheduler

    elif model_name == "SIMPLE" or model_name == "SIMPLE_BI":
        args.layers = 6 if args.layers <= 0 else args.layers
        is_bidirectional = model_name == "SIMPLE_BI"
        model = BaseLines.SimpleTransformer(vocab_size=len(tokenizer), num_layers=args.layers,
                                            ntokens=hp.input_size,
                                            bidirectional=is_bidirectional)
        metrics = BasicTransductionMetrics()
        optimizer = AdamW(model.parameters(), lr=hp.learning_rate, betas=(0.9, 0.98), weight_decay=1e-5)
        if not args.no_optimizer:
            optimizer = TrainUtil.NoamOptim(optimizer, model.d_model)  # Use sqrt lr scheduler
    else:
        args.layers = 1
        model = BaseLines.Seq2SeqBiLSTM(vocab_size=len(tokenizer), input_dim=256, hidden_dim=1024, num_layers=1)
        metrics = BasicLSTMTransductionMetrics(model, tokenizer)
        optimizer = AdamW(model.parameters(), lr=hp.learning_rate, betas=(0.9, 0.98), weight_decay=1e-5)
        if not args.no_optimizer:
            optimizer = TrainUtil.NoamOptim(optimizer, model.d_model)

    # Set Save Path
    save_path = os.path.join(
        "Bracket",
        training_set_name.replace(".json", ""),
        args.tokenize,
        "layer-count-" + str(args.layers),
        model_name + ("" if not args.no_pretrained_weight else "-scratch")
    )
    trainer.save_dir = save_path

    # Load model weights if desired
    if args.init_from:
        load_path = os.path.join(
            "Bracket",
            args.init_from.replace(".json", ""),
            args.tokenize,
            "layer-count-" + str(args.layers),
            model_name + ("" if not args.no_pretrained_weight else "-scratch")
        )
        former_save_dir = trainer.save_dir
        trainer.save_dir = load_path
        trainer.load_state_dict()

        # Load best checkpoint if possible
        saved_best = trainer.load_state_dict()
        if saved_best:
            print(f"Loaded parameters of model trained on {args.init_from}.")
            model.load_state_dict(saved_best, False)
        else:
            print("Couldn't find saved model. Training new model instead.")
        trainer.save_dir = former_save_dir

    # Initialize criterion
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    criterion.to(hp.device)

    if args.load:
        try:
            saved_best = trainer.load_state_dict()
            model.load_state_dict(saved_best, False)
            print("Loaded model from dir: " + trainer.save_dir)
        except Exception as err:
            print("Error occurred. Building new optimizer/model instead.")
            print("Error: " + str(err))

    model.to(device=hp.device)

    if args.model_tag:
        trainer.save_dir = os.path.join(trainer.save_dir, args.model_tag)


    if not args.eval and not args.eval_from:
        # Start training
        trainer.train(model=model, optimizer=optimizer, tokenizer=tokenizer, criterion=criterion, data_set=train_set,
                      metrics=metrics)

    # Load best checkpoint if possible
    saved_best = trainer.load_state_dict()
    if saved_best:
        print("Loaded saved model for evaluation.")
        model.load_state_dict(saved_best, False)
    else:
        print("Couldn't find saved model. Evaluating with last checkpoint.")

    if model_name == "BART":
        metrics = TransductionMetrics()
    elif model_name == "SIMPLE" or model_name == "SIMPLE_BI":
        metrics = TransductionMetrics()
    else:
        metrics = LSTMTransductionMetrics(model, tokenizer)

    # Evaluate
    trainer.test(model=model, optimizer=optimizer, tokenizer=tokenizer, criterion=criterion, data_set=test_set,
                 metrics=metrics,
                 split_into_two=False)


def calculate_edit_distance(labels, predicted):
    index = random.randrange(labels.size(dim=0))
    return textdistance.levenshtein.distance(labels[index, :].tolist(), predicted[index, :].tolist()) * labels.size(
        dim=0) / labels.size(dim=1)


def brackets_only(s: str):
    return re.findall(bracket_pattern, s)


def bracket_edit_distance(label, prediction):
    return textdistance.levenshtein.distance(brackets_only(label), brackets_only(prediction))


def bracket_accuracy(label, prediction):
    label = brackets_only(label)
    prediction = brackets_only(prediction)

    if len(prediction) < len(label):
        for i in range(len(label) - len(prediction)):
            prediction.append("<pad>")
    prediction = prediction[0: len(label)]
    return accuracy_score(label, prediction)


def calculate_edit_distance_all(labels, predicted):
    pairs = zip(labels.tolist(), predicted.tolist())
    return np.sum([textdistance.levenshtein.distance(entry[0], entry[1]) for entry in pairs])


def count_full_hit_percentage(labels, predicted):
    non_full_hits = torch.count_nonzero(torch.count_nonzero(labels - predicted, dim=1))
    num_full_hits = labels.size(0) - non_full_hits
    return num_full_hits.item() / labels.size(0)


class TransductionMetrics(TrainUtil.Metrics):

    def __init__(self, reduced=False):
        super().__init__()
        self.metrics = np.zeros(5)
        self.errors = []
        self.log_num = 0
        self.current_prefix = "<init>"

    def prepare_data(self, batch, outputs):
        input_ids, in_masks, l_ids, l_masks = (tensor.cpu().detach() for tensor in batch)
        outputs = outputs.cpu().detach()

        # Adapt
        l_ids = l_ids[:, 1:]
        l_masks = l_masks[:, 1:]
        predicted_indices = torch.argmax(outputs, dim=-1)
        return input_ids, in_masks, l_ids, l_masks, predicted_indices

    def update(self, batch, outputs):

        input_ids, in_masks, labels, l_masks, predicted_indices = self.prepare_data(batch, outputs)

        # Sample one erroneous prediction
        error_indices = torch.nonzero(torch.any(predicted_indices != labels, dim=1)).squeeze().squeeze().tolist()
        error_indices = [error_indices] if not isinstance(error_indices, list) else error_indices
        if error_indices and len(error_indices) > 0:
            idx = error_indices[random.randrange(len(error_indices))]
            self.errors.append((input_ids[idx], labels[idx], predicted_indices[idx]))

        assert labels.size() == predicted_indices.size(), f"{labels.size()} != {predicted_indices.size()}"

        # Calculate metrics
        avg_length = torch.count_nonzero(in_masks) / in_masks.size(0)
        full_hit_perc = count_full_hit_percentage(labels=labels,
                                                  predicted=predicted_indices.reshape(labels.size()))  # full hits

        # mask and flatten
        mask = l_masks.reshape(-1)
        labels = labels.reshape(-1).masked_select(mask == 1)
        predicted_indices = predicted_indices.reshape(-1).masked_select(mask == 1)

        labels = labels.reshape(-1, 1)
        predicted_indices = predicted_indices.reshape(-1, 1)

        # Calculate residual
        acc = accuracy_score(y_true=labels, y_pred=predicted_indices)  # accuracy,
        f1 = f1_score(y_true=labels, y_pred=predicted_indices, average="micro")  # f1
        edit_dist = calculate_edit_distance(labels, predicted_indices)  # levenshtein

        # Update
        self.metrics += np.array([avg_length, acc, f1, edit_dist, full_hit_perc])

    def print(self, index, set_size, prefix, count, epoch, loss, decoder=None):

        if not self.current_prefix == prefix:
            self.log_num = 0
            self.current_prefix = prefix

        avg_loss = loss / count
        metrics = self.metrics / count
        [avg_length, accuracy, f1, levenshtein, full_hit_perc] = metrics
        print(
            f"Batch {index}/{set_size} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Lengths : {avg_length:.2f}")

        bracket_acc = 1.0
        if len(self.errors) > 0:
            error = self.errors[random.randrange(len(self.errors))]
            inp, label, pred = (decoder(ids) for ids in error)
            print(f"IN: \n\t{inp} \nOUT: \n\t{pred} \nTRUE: \n\t{label}\n")

            bracket_acc = 0.
            for error in self.errors:
                inp, label, pred = (decoder(ids) for ids in error)
                bracket_acc += bracket_accuracy(label, pred)
            bracket_acc /= len(self.errors)

        wandb.log({
            prefix + "in_lengths": avg_length,
            prefix + "epoch": epoch,
            prefix + "loss": avg_loss,
            prefix + "accuracy": accuracy,
            prefix + "f1": f1,
            prefix + "levenshtein": levenshtein,
            prefix + "full_hit%": full_hit_perc,
            prefix + "bracket_acc": bracket_acc,
            "custom_step": self.log_num
        })

        self.log_num += 1

    def reset(self):
        self.metrics = np.zeros(5)
        self.errors = []


class BasicTransductionMetrics(TransductionMetrics):

    def __init__(self):
        super().__init__()
        self.acc = 0
        self.current_prefix = ""
        self.log_num = 0

    def prepare_data(self, batch, outputs):
        input_ids, in_masks, l_ids, l_masks = (tensor for tensor in batch)
        outputs = outputs

        # Adapt
        l_ids = l_ids[:, 1:]
        l_masks = l_masks[:, 1:]

        predicted_indices = torch.argmax(outputs, dim=-1)
        return input_ids, in_masks, l_ids, l_masks, predicted_indices

    def update(self, batch, outputs):
        with torch.no_grad():
            input_ids, in_masks, labels, l_masks, predicted_indices = self.prepare_data(batch, outputs)
            self.acc += ((labels == predicted_indices).sum().item() / labels.numel())  # acc

    def print(self, index, set_size, prefix, count, epoch, loss, decoder=None):
        if not self.current_prefix == prefix:
            self.log_num = 0
            self.current_prefix = prefix

        avg_loss = loss / count
        avg_acc = self.acc / count

        print(
            f"Batch {index}/{set_size} - Loss: {avg_loss:.4f}, Accuracy / F1: {avg_acc:.4f}")

        wandb.log({
            prefix + "epoch": epoch,
            prefix + "loss": avg_loss,
            prefix + "accuracy": avg_acc,
            prefix + "f1": avg_acc,
            "custom_step": self.log_num
        })

        self.log_num += 1
        self.reset()

    def reset(self):
        self.acc = 0


class TransductionTrainer(TrainUtil.Trainer):
    def generate(self, model: BartForConditionalGeneration, batch):
        in_ids, in_masks, l_ids, l_masks = batch
        with torch.no_grad():
            outputs = model.generate(input_ids=in_ids, max_length=l_ids.size(1) + 2)
        return outputs

    def __init__(self, hp):
        super().__init__(hp)

    def decode_tokens(self, tokenizer: BartTokenizer, ids) -> str | list[str]:
        if len(ids.size()) == 1:
            return tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=True)
        return [tokenizer.decode(ids[i], skip_special_tokens=False, clean_up_tokenization_spaces=True) for i in
                range(ids.size(0))]

    def encode(self, inputs, labels, tokenizer, return_entry_ids=False):
        in_ids = []
        in_masks = []
        l_ids = []
        l_masks = []
        entry_ids = []

        filtered_count = 0
        max_length = self.hp.input_size if self.hp.input_size > 0 else tokenizer.model_max_length

        for idx, (i, l) in enumerate(zip(inputs, labels)):
            encoded_i = tokenizer.encode_plus(i, add_special_tokens=True, padding='max_length',
                                              max_length=max_length,
                                              return_attention_mask=True, return_tensors='pt')

            encoded_l = tokenizer.encode_plus(l, add_special_tokens=True, padding='max_length',
                                              max_length=max_length,
                                              return_attention_mask=True, return_tensors='pt')

            if encoded_i['input_ids'].size(dim=1) <= max_length and encoded_l['input_ids'].size(dim=1) <= max_length:
                in_ids.append(encoded_i['input_ids'])
                in_masks.append(encoded_i['attention_mask'])
                l_ids.append(encoded_l['input_ids'])
                l_masks.append(encoded_l['attention_mask'])
                entry_ids.append(idx)
            else:
                filtered_count += 1

        # Print filtered
        print(f"Filtered {filtered_count} input-label-pairs as they exceeded max length {max_length}.")

        # Transform to tensors
        in_ids = torch.cat(in_ids, dim=0)
        in_masks = torch.cat(in_masks, dim=0)
        l_ids = torch.cat(l_ids, dim=0)
        l_masks = torch.cat(l_masks, dim=0)

        if return_entry_ids:
            return entry_ids, in_ids, in_masks, l_ids, l_masks
        return in_ids, in_masks, l_ids, l_masks

    def forward_to_model(self, model: BartForConditionalGeneration, criterion, batch):
        in_ids, in_masks, l_ids, l_masks = batch
        outputs = model(input_ids=in_ids, attention_mask=in_masks,
                        decoder_input_ids=l_ids[:, :-1].contiguous(),
                        decoder_attention_mask=l_masks[:, :-1].contiguous(),
                        labels=l_ids[:, 1:].contiguous())

        loss = criterion(outputs.logits.reshape(-1, outputs.logits.shape[-1]), l_ids[:, 1:].reshape(-1))
        return outputs.logits, loss


class SimpleTransductionTrainer(TransductionTrainer):
    def forward_to_model(self, model: BaseLines.SimpleTransformer, criterion: CrossEntropyLoss, batch):
        in_ids, in_masks, l_ids, l_masks = batch
        outputs = model(in_ids=in_ids, in_masks=in_masks,
                        l_ids=l_ids[:, :-1].contiguous(),
                        l_masks=l_masks[:, :-1].contiguous())

        loss = criterion(outputs.reshape(-1, outputs.shape[-1]), l_ids[:, 1:].reshape(-1))
        # loss_masked = loss.masked_select(l_masks[:, 1:].reshape(-1) == 1)  # Only compute loss for non-special tokens
        return outputs, loss  # loss_masked.mean()


class BasicLSTMTransductionMetrics(BasicTransductionMetrics):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.generator = model.generator

    def prepare_data(self, batch, outputs: torch.Tensor):
        # Find nearest
        with torch.no_grad():
            predicted_indices = self.generator(outputs)
            predicted_indices = torch.argmax(predicted_indices, dim=-1)

            # Move to CPU
            input_ids, in_masks, labels, l_masks = (tensor for tensor in batch)

            diff_size = labels.size(1) - 1 - predicted_indices.size(1)
            if diff_size > 0:
                pad_id = self.tokenizer.pad_token_id
                batch_size = labels.size(0)
                size = torch.Size((batch_size, diff_size))
                predicted_indices = torch.cat((predicted_indices, torch.full(size, pad_id, device=predicted_indices.get_device())), dim=1)

            return input_ids, in_masks, labels[:, 1:], l_masks[:, 1:], predicted_indices


class LSTMTransductionMetrics(TransductionMetrics):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.generator = model.generator

    def prepare_data(self, batch, outputs):
        # Find nearest
        with torch.no_grad():
            predicted_indices = self.generator(outputs)
            predicted_indices = torch.argmax(predicted_indices, dim=-1)

        # Move to CPU
        input_ids, in_masks, labels, l_masks = (tensor.cpu().detach() for tensor in batch)
        predicted_indices = predicted_indices.cpu().detach()

        diff_size = labels.size(1) - 1 - predicted_indices.size(1)
        if diff_size > 0:
            pad_id = self.tokenizer.pad_token_id
            batch_size = labels.size(0)
            size = torch.Size((batch_size, diff_size))
            predicted_indices = torch.cat((predicted_indices, torch.full(size, pad_id)), dim=1)

        return input_ids, in_masks, labels[:, 1:], l_masks[:, 1:], predicted_indices


class Seq2SeqLSTMTrainer(TransductionTrainer):
    def __init__(self, hp: Common.HyperParams, tokenizer: BartTokenizer):
        super(Seq2SeqLSTMTrainer, self).__init__(hp)
        self.tokenizer = tokenizer

    def generate(self, model: BaseLines.Seq2SeqBiLSTM, batch):
        in_ids, in_masks, l_ids, l_masks = batch
        pad_id = self.tokenizer.pad_token_id
        sos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id
        seq_length = in_ids.size(1)

        def num_pad_token(t):
            return torch.eq(t, pad_id).sum().item()

        def true_lengths(ids):
            return torch.tensor(
                [seq_length - num_pad_token(ids[b, :]) for b in range(ids.size(0))]
            )

        # Calculate Lengths
        in_len = true_lengths(in_ids)
        l_len = true_lengths(l_ids)
        longest_len = torch.max(l_len).item()

        return model.generate(in_ids, in_masks, in_len, longest_len, sos_id=sos_id, eos_id=eos_id)

    def decode_tokens(self, tokenizer: BartTokenizer, ids) -> str | list[str]:
        if len(ids.size()) == 1:
            return tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=True)
        return [tokenizer.decode(ids[i], skip_special_tokens=False, clean_up_tokenization_spaces=True) for i in
                range(ids.size(0))]

    def encode(self, inputs, labels, tokenizer):
        self.tokenizer = tokenizer
        r = super().encode(inputs, labels, tokenizer)
        return r

    def forward_to_model(self, model: BaseLines.Seq2SeqBiLSTM, criterion, batch):
        in_ids, in_masks, l_ids, l_masks = batch
        pad_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        seq_length = in_ids.size(1)

        def num_pad_token(t):
            return torch.eq(t, pad_id).sum().item()

        def true_lengths(ids):
            return torch.tensor(
                [seq_length - num_pad_token(ids[b, :]) for b in range(ids.size(0))]
            )

        # Calculate Lengths
        in_len = true_lengths(in_ids)
        l_len = true_lengths(l_ids) - 1  # subtract one as decoder inputs will be shifted by one

        max_len = torch.max(l_len).item()

        # Pass and return
        states, hidden, pre_out = model(in_ids, in_masks,
                                        l_ids[:, :-1].contiguous(),
                                        l_masks[:, :-1].contiguous(),
                                        in_len, l_len)
        prediction = model.generator(pre_out)
        loss = criterion(prediction.contiguous().view(-1, prediction.size(-1)),
                         l_ids[:, 1: max_len + 1].contiguous().view(-1))
        return pre_out, loss


if __name__ == "__main__":
    print("Starting script...")
    start()
