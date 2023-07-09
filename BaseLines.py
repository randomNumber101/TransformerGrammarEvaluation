import random
from typing import Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import one_hot
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence


class TokenEmbedder:
    def __init__(self):
        self.entry_to_id = dict()
        self.id_to_entry = dict()
        self.next_id = 0

    def get_id(self, entry):
        if not entry in self.entry_to_id:
            self.entry_to_id[entry] = self.next_id
            self.id_to_entry[self.next_id] = entry
            self.next_id += 1
        return self.entry_to_id[entry]

    def get_entry(self, id):
        return self.id_to_entry.get(id)

    def encode_single(self, entry, num_classes, device):
        return one_hot(torch.tensor(self.get_id(entry), device=device))

    def decode_single(self, one_hot_id):
        id = torch.argmax(one_hot_id, dim=1)
        return self.entry_to_id.get(id)

    def encode(self, t: Tensor, num_classes=-1, device="cpu"):
        data = [self.encode_single(old_id, num_classes, device) for old_id in t.view(-1)]
        newShape = torch.Size(list(t.shape) + [num_classes])
        return torch.cat(data).view(newShape)

    def decode(self, t: Tensor, device="cpu"):
        orig_shape = t.size()[:-1]
        flattened = t.view(-1, t.size(dim=-1))
        return torch.tensor([self.decode_single(ohv) for ohv in flattened], device=device).reshape(orig_shape)

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)


def getLastSequenceElement(t):
    if isinstance(t, Tensor):
        return t[:, -1, :]
    elif isinstance(t, PackedSequence):
        lengths = torch.ones(t.data.size(1), device="cpu")
        last_seq_entries = torch.tensor([t[i, length, :] for i, length in enumerate(t.batch_sizes)])
        return pack_padded_sequence(last_seq_entries.unsqueeze(1), lengths)


def get_tensor(t: Union[torch.Tensor, PackedSequence]) -> Tensor:
    if isinstance(t, PackedSequence):
        return t.data
    else:
        return t


class Seq2SeqBiLSTM(nn.Module):

    class Generator(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(Seq2SeqBiLSTM.Generator, self).__init__()
            self.output_fc = nn.Linear(input_dim, output_dim)
            self.softmax = nn.Softmax(dim=2)

        def forward(self, hidden):
            if len(hidden.shape) == 2:
                hidden = hidden.unsqueeze(0)  # Expecting batched input
            out = self.output_fc(hidden)
            out = self.softmax(out)
            return out

        def to(self, device):
            self.output_fc.to(device)
            self.softmax.to(device)

    def __init__(self, vocab_size, input_dim, hidden_dim, device="cpu", teacher_forcing_ratio=1.0):
        super(Seq2SeqBiLSTM, self).__init__()
        self.device = device
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.input_dim = input_dim
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        # Input embedding
        self.enc_embedding = nn.Embedding(vocab_size, input_dim)
        self.dec_embedding = nn.Embedding(vocab_size, input_dim)

        # Encoder + Reducer
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True, device=device)
        self.fc_reduce_hidden = nn.Linear(hidden_dim * 2, hidden_dim, device=device)  # reduces the hidden state of encoder
        self.fc_reduce_cell = nn.Linear(hidden_dim * 2, hidden_dim, device=device)  # reduces the cell state of encoder

        # Decoder Cell
        self.decoder = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=False, device=device)  # single layer decoder
        self.fc_predict = nn.Linear(hidden_dim, input_dim, device=device)  # produces hidden input vectors for decoder

        # Output
        self.generator = Seq2SeqBiLSTM.Generator(input_dim, vocab_size)

    # TODO: Fix PackedSequence Processing
    def forward(self, src: Union[torch.Tensor, PackedSequence], targets: Union[torch.Tensor, PackedSequence, int]):

        # Move to device
        (src, targets) = (src.to(self.device), targets.to(self.device))
        self.to(self.device)

        if isinstance(targets, int):
            target_len = targets
        else:
            tgt_tensor = get_tensor(targets)
            target_len = tgt_tensor.size(1)

        batch_size = get_tensor(src).size(0)
        outputs = torch.zeros(target_len, batch_size, self.input_dim, device=self.device)

        # Embed input
        src = self.enc_embedding(src)
        targets = self.dec_embedding(targets)

        # forward encoder
        enc_out, enc_hidden = self.encoder(src)

        # concatenate hidden states and cell states and reduce them

        (hn, cn) = enc_hidden
        (hn, cn) = (
            torch.cat((hn[0], hn[1]), dim=1),
            torch.cat((cn[0], cn[1]), dim=1))  # reshape to (batch_size, 2 * hidden_dim)
        (hn, cn) = (self.fc_reduce_hidden(hn), self.fc_reduce_cell(cn))
        (hn, cn) = (hn.unsqueeze(0), cn.unsqueeze(0))

        # decoder ins
        dec_input = get_tensor(src)[:, -1, :]  # last sequence chars, shape (batch_size, 1, input_dim)
        dec_hidden = (hn, cn)  # will contain final states for fwd and bckwd pass (dim = 2 * hidden_dim)

        if isinstance(targets, int) or random.random() > self.teacher_forcing_ratio:
            # auto regressive
            for t in range(target_len):
                dec_output, dec_hidden = self.decoder(dec_input.unsqueeze(1), dec_hidden)
                prediction_hidden = self.fc_predict(dec_output.squeeze(1))
                outputs[t] = prediction_hidden
                dec_input = prediction_hidden
        else:
            # teacher forcing
            for t in range(target_len):
                dec_output, dec_hidden = self.decoder(dec_input.unsqueeze(1), dec_hidden)
                prediction_hidden = self.fc_predict(dec_output.squeeze(1))
                outputs[t] = prediction_hidden
                dec_input = targets[:, t, :]

        return outputs.permute(1, 0, 2)  # put batch at start

    def to(self, device):
        super(Seq2SeqBiLSTM, self).to(device)
        self.device = device

        self.enc_embedding.to(device=device)
        self.dec_embedding.to(device=device)

        self.encoder.to(device=device)
        self.fc_reduce_hidden.to(device=device)
        self.fc_reduce_cell.to(device=device)

        self.decoder.to(device=device)
        self.fc_predict.to(device=device)
        self.generator.to(device=device)





def test():
    vocab_size = 50000
    model = Seq2SeqBiLSTM(vocab_size=vocab_size, input_dim=128, hidden_dim=128)
    x = model(torch.randint(vocab_size, (4, 3)), torch.randint(vocab_size, (4, 10)))
    print(x.size())