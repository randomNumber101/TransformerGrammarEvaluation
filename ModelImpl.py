import math
import random
from typing import Union

import torch
from torch import Tensor, nn as nn
from torch.nn import functional as F
from torch.nn.functional import one_hot
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence


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


# from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = torch.permute(x, (1, 0, 2))
        x = x + self.pe[:x.size(0)]
        x = torch.permute(x, (1, 0, 2))
        return self.dropout(x)


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


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.1, device="cpu"):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True,
                            device=device)

    def forward(self, x, mask, lengths):
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        output, final = self.lstm(packed)
        output, _ = pad_packed_sequence(output, batch_first=True, total_length=x.size(1))
        return output, final


class Decoder(nn.Module):
    """A conditional RNN decoder with attention."""

    def __init__(self, input_size, hidden_size, attention, num_layers=1, dropout=0.5):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention = attention
        self.dropout = dropout

        self.rnn = nn.LSTM(input_size + 2 * hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

        # to initialize from the final encoder state
        self.bridge_hn = nn.Linear(2 * hidden_size, hidden_size, bias=True)
        self.bridge_cn = nn.Linear(2 * hidden_size, hidden_size, bias=True)

        self.dropout_layer = nn.Dropout(p=dropout)
        self.pre_output_layer = nn.Linear(hidden_size + 2 * hidden_size + input_size,
                                          hidden_size, bias=False)

    def forward_step(self, prev_embed, encoder_hidden, src_mask, proj_key, hidden):
        """Perform a single decoder step (1 word)"""

        # compute context vector using attention mechanism
        (hn, cn) = hidden
        (hn, cn) = (hn[0], cn[0])
        query = hn.unsqueeze(1)  # [B, D] -> [B, 1, D]
        context, attn_probs = self.attention(
            query=query, proj_key=proj_key,
            value=encoder_hidden, mask=src_mask)

        # update rnn hidden state
        rnn_input = torch.cat([prev_embed, context], dim=2)
        output, hidden = self.rnn(rnn_input, hidden)

        pre_output = torch.cat([prev_embed, output, context], dim=2)
        pre_output = self.dropout_layer(pre_output)
        pre_output = self.pre_output_layer(pre_output)

        return output, hidden, pre_output

    def forward(self, trg_embed, encoder_hidden, encoder_final,
                src_mask, trg_mask, trg_lengths, sos=None, eos=None, hidden=None, max_len=None):
        """Unroll the decoder one step at a time."""

        # the maximum number of steps to unroll the RNN
        if max_len is None:
            max_len = torch.max(trg_lengths).item()

        # initialize decoder hidden state
        if hidden is None:
            hidden = self.init_hidden(encoder_final)

        # pre-compute projected encoder hidden states
        # (the "keys" for the attention mechanism)
        # this is only done for efficiency
        proj_key = self.attention.key_layer(encoder_hidden)

        # here we store all intermediate hidden states and pre-output vectors
        decoder_states = []
        pre_output_vectors = []

        # unroll the decoder RNN for max_len steps
        for i in range(max_len):
            prev_embed = trg_embed[:, i].unsqueeze(1)
            output, hidden, pre_output = self.forward_step(
                prev_embed, encoder_hidden, src_mask, proj_key, hidden)
            decoder_states.append(output)
            pre_output_vectors.append(pre_output)

        decoder_states = torch.cat(decoder_states, dim=1)
        pre_output_vectors = torch.cat(pre_output_vectors, dim=1)
        return decoder_states, hidden, pre_output_vectors  # [B, MAX_N, D]

    def generate(self, encoder_hidden, encoder_final, src_mask, max_len, sos_id, generator, embedder=None, hidden=None,
                 eos_id=None, device="cuda"):
        # initialize decoder hidden state
        if hidden is None:
            hidden = self.init_hidden(encoder_final)

        # pre-compute projected encoder hidden states
        # (the "keys" for the attention mechanism)
        # this is only done for efficiency
        proj_key = self.attention.key_layer(encoder_hidden)

        # Initialize trgts
        batch_size = encoder_hidden.size(0)
        trg_embed = torch.ones(batch_size, 1, dtype=torch.long, device=device).fill_(sos_id)
        if embedder:
            trg_embed = embedder(trg_embed)

        # First token will always be SOS
        output_ids = [torch.ones(batch_size, dtype=torch.long, device=device).fill_(sos_id)]

        # unroll the decoder RNN for max_len steps
        for i in range(max_len):
            prev_embed = trg_embed[:, -1].unsqueeze(1)
            output, hidden, pre_output = self.forward_step(
                prev_embed, encoder_hidden, src_mask, proj_key, hidden)

            prob = generator(pre_output)
            pred = torch.max(prob, dim=-1).indices
            output_ids.append(pred.squeeze(1))
            if embedder:
                pred = embedder(pred)
            trg_embed = torch.cat((trg_embed, pred), dim=1)

        output_ids = torch.stack(output_ids, dim=1)
        return output_ids

    def init_hidden(self, encoder_final):
        """Returns the initial decoder state,
        conditioned on the final encoder state."""

        if encoder_final is None:
            return None  # start with zeros

        (hn_o, cn_o) = encoder_final

        processed_hn = []
        processed_cn = []
        for i in range(self.num_layers):
            (hn, cn) = (
                torch.cat((hn_o[i], hn_o[i + 1]), dim=1),
                torch.cat((cn_o[i + 1], cn_o[i + 1]), dim=1))  # reshape to (batch_size, 2 * hidden_dim)
            (hn, cn) = (self.bridge_hn(hn), self.bridge_cn(cn))
            (hn, cn) = (torch.tanh(hn), torch.tanh(cn))
            processed_hn.append(hn)
            processed_cn.append(cn)

        return torch.stack(processed_hn, dim=0), torch.stack(processed_cn, dim=0)


class BahdanauAttention(nn.Module):
    """Implements Bahdanau (MLP) attention"""

    def __init__(self, hidden_size, key_size=None, query_size=None):
        super(BahdanauAttention, self).__init__()

        # We assume a bi-directional encoder so key_size is 2*hidden_size
        key_size = 2 * hidden_size if key_size is None else key_size
        query_size = hidden_size if query_size is None else query_size

        self.key_layer = nn.Linear(key_size, hidden_size, bias=False)
        self.query_layer = nn.Linear(query_size, hidden_size, bias=False)
        self.energy_layer = nn.Linear(hidden_size, 1, bias=False)

        # to store attention scores
        self.alphas = None

    def forward(self, query=None, proj_key=None, value=None, mask=None):
        assert mask is not None, "mask is required"

        # We first project the query (the decoder state).
        # The projected keys (the encoder states) were already pre-computated.
        query = self.query_layer(query)

        # Calculate scores.
        scores = self.energy_layer(torch.tanh(query + proj_key))
        scores = scores.squeeze(2).unsqueeze(1)

        # Mask out invalid positions.
        # The mask marks valid positions so we invert it using `mask & 0`.
        scores.data.masked_fill_(mask.unsqueeze(1) == 0, -float('inf'))

        # Turn scores to probabilities.
        alphas = F.softmax(scores, dim=-1)
        self.alphas = alphas

        # The context vector is the weighted sum of the values.
        context = torch.bmm(alphas, value)

        # context shape: [B, 1, 2D], alphas shape: [B, 1, M]
        return context, alphas


class Generator(nn.Module):
    """Define standard linear + softmax generation step."""

    def __init__(self, hidden_size, vocab_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, x):
        return F.softmax(self.proj(x), dim=-1)


class AvgClassificationHead(nn.Module):
    def __init__(self, feature_dim, class_count):
        super(AvgClassificationHead, self).__init__()
        self.fc = nn.Linear(feature_dim, class_count)
        self.dropout = nn.Dropout(p=0.1)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, features):
        features = self.dropout(features)
        avg = torch.mean(features, dim=1)
        avg = self.fc(avg)
        avg = self.softmax(avg)
        return avg


class AttentionClassificationHead(nn.Module):
    def __init__(self, hiden_dim, class_count):
        super(AttentionClassificationHead, self).__init__()
        self.attention = BahdanauAttention(hiden_dim, query_size=hiden_dim * 2)
        self.preprocessor = nn.Linear(hiden_dim * 2, hiden_dim * 2, bias=True)
        self.dropout = nn.Dropout(p=0.1, inplace=False)
        self.predictor = nn.Linear(hiden_dim * 2, class_count, bias=True)

    def forward(self, features, mask):
        proj_key = self.attention.key_layer(features)
        query = torch.mean(features, dim=1).unsqueeze(1)
        context, attn_probs = self.attention(
            query=query, proj_key=proj_key,
            value=features, mask=mask
        )

        y = self.preprocessor(context.squeeze())
        y = self.dropout(y)
        y = self.predictor(y)
        y = F.softmax(y, dim=-1)
        return y


class SimpleClassificationHead(nn.Module):
    def __init__(self, input_dim, class_count, bidirectional=True):
        super(SimpleClassificationHead, self).__init__()
        self.fc = nn.Linear(input_dim * (2 if bidirectional else 1), class_count)

    def forward(self, last_hidden):
        y = self.fc(last_hidden)
        return F.softmax(y, dim=-1)


class LegacySeq2SeqBiLSTM(nn.Module):
    def __init__(self, vocab_size, input_dim, hidden_dim, device="cpu", teacher_forcing_ratio=1.0):
        super(LegacySeq2SeqBiLSTM, self).__init__()
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
        self.fc_reduce_hidden = nn.Linear(hidden_dim * 2, hidden_dim,
                                          device=device)  # reduces the hidden state of encoder
        self.fc_reduce_cell = nn.Linear(hidden_dim * 2, hidden_dim, device=device)  # reduces the cell state of encoder

        # Decoder Cell
        self.decoder = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=False,
                               device=device)  # single layer decoder
        self.fc_predict = nn.Linear(hidden_dim, input_dim, device=device)  # produces hidden input vectors for decoder

        # Output
        self.generator = Generator(input_dim, vocab_size)

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
