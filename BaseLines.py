import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch import nn as nn
from torch.nn import Transformer
from transformers.models.bart.modeling_bart import BartClassificationHead

from ModelImpl import LegacySeq2SeqBiLSTM, BahdanauAttention, Encoder, Decoder, Generator, AttentionClassificationHead, \
    PositionalEncoding

from transformers import BartForSequenceClassification


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size: int, ntokens=512, d_model=512, num_layers=6, bidirectional=False, device="cpu"):
        super().__init__()
        self.d_model = d_model
        self.src_embed = nn.Embedding(vocab_size, self.d_model)
        self.tgt_embed = nn.Embedding(vocab_size, self.d_model)
        self.positional_encoder = PositionalEncoding(d_model=self.d_model, max_len=ntokens)
        self.model = Transformer(d_model=self.d_model, batch_first=True, num_encoder_layers=num_layers,
                                 num_decoder_layers=num_layers)
        self.bidirectional = bidirectional
        self.generator = Generator(hidden_size=self.d_model, vocab_size=vocab_size)
        self.device = device

    def forward(self, in_ids, l_ids, in_masks, l_masks):
        in_ids = self.src_embed(in_ids.long()) * math.sqrt(self.d_model)  # scale by sqrt of dmodel
        in_ids = self.positional_encoder(in_ids)

        l_ids = self.tgt_embed(l_ids.long()) * math.sqrt(self.d_model)
        l_ids = self.positional_encoder(l_ids)

        # Create Masks
        src_seq_len = in_ids.size(1)
        tgt_seq_len = l_ids.size(1)
        src_mask = torch.zeros(src_seq_len, src_seq_len, device=self.device).type(torch.bool)
        if not self.bidirectional:
            tgt_mask = torch.triu(torch.full((tgt_seq_len, tgt_seq_len), float('-inf'), device=self.device), diagonal=1)
        else:
            tgt_mask = torch.zeros(tgt_seq_len, tgt_seq_len, device=self.device).type(torch.bool)
        in_masks = in_masks == 0.0
        l_masks = l_masks == 0.0

        out = self.model(src=in_ids, tgt=l_ids,
                         src_mask=src_mask, tgt_mask=tgt_mask,
                         src_key_padding_mask=in_masks,
                         tgt_key_padding_mask=l_masks)
        return self.generator(out)

    def to(self, device):
        super(SimpleTransformer, self).to(device)
        self.device = device


class SimpleClassifier(nn.Module):

    def __init__(self, vocab_size, num_classes, hidden_dim=512, num_layers=3, dropout=0.1,
                 ntoken=512, device="cpu", nhead=8):
        super(SimpleClassifier, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model=hidden_dim, max_len=ntoken)
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=num_layers
        )
        self.classify = BartClassificationHead(hidden_dim, hidden_dim, num_classes, dropout)
        self.device = device

    def forward(self, input_ids, mask):

        mask = mask == 0.0

        x = self.embedding(input_ids)
        x = self.pos_encoder(x)
        x = self.encoder(x, src_key_padding_mask=mask)
        x = torch.mean(x, dim=1)
        x = self.classify(x)
        return torch.softmax(x, dim=-1)


class Seq2SeqBiLSTM(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models_data.
    """

    def __init__(self, vocab_size, input_dim=256, hidden_dim=1024, num_layers=1, dropout=0.2, device="cpu"):
        super(Seq2SeqBiLSTM, self).__init__()
        self.d_model = hidden_dim
        self.attention = BahdanauAttention(hidden_dim)
        self.src_embed = nn.Embedding(vocab_size, input_dim)
        self.trg_embed = nn.Embedding(vocab_size, input_dim)
        self.encoder = Encoder(input_dim, hidden_dim, num_layers=num_layers, dropout=dropout)
        self.decoder = Decoder(input_dim, hidden_dim, self.attention, num_layers=num_layers, dropout=dropout)
        self.generator = Generator(hidden_dim, vocab_size)
        self.device = device

    def forward(self, src, src_mask, trg, trg_mask, src_lengths, trg_lengths):
        """Take in and process masked src and target sequences."""
        encoder_hidden, encoder_final = self.encode(src, src_mask, src_lengths)
        return self.decode(encoder_hidden, encoder_final, src_mask, trg, trg_mask, trg_lengths)

    def encode(self, src, src_mask, src_lengths):
        return self.encoder(self.src_embed(src), src_mask, src_lengths)

    def decode(self, encoder_hidden, encoder_final, src_mask, trg, trg_mask, trg_lengths,
               decoder_hidden=None):
        return self.decoder(self.trg_embed(trg), encoder_hidden, encoder_final,
                            src_mask, trg_mask, trg_lengths, hidden=decoder_hidden)

    def generateLeg(self, src, src_mask, src_lengths, max_len, sos_id, eos_id):
        (src, src_mask) = (src.to(self.device), src_mask.to(self.device))
        encoder_hidden, encoder_final = self.encode(src, src_mask, src_lengths)
        return self.decoder.generate(encoder_hidden, encoder_final, src_mask, max_len,
                                     generator=self.generator, embedder=self.trg_embed,
                                     sos_id=sos_id, eos_id=eos_id, device=self.device)

    def generate(self, src, src_mask, src_lengths, max_len, sos_id, eos_id):
        batch_size = src.size(0)

        with torch.no_grad():
            encoder_hidden, encoder_final = self.encode(src, src_mask, src_lengths)
            prev_y = torch.ones(batch_size, 1, dtype=torch.long, device=self.device).fill_(sos_id)
            prev_y = self.trg_embed(prev_y)
            trg_mask = torch.ones_like(prev_y)

        output = []
        # attention_scores = []
        hidden = None

        for i in range(max_len):
            with torch.no_grad():
                out, hidden, pre_output = self.decoder(
                    prev_y, encoder_hidden, encoder_final, src_mask,
                    trg_mask, hidden, max_len=prev_y.size(1))

                # we predict from the pre-output layer, which is
                # a combination of Decoder state, prev emb, and context
                prob = self.generator(pre_output[:, -1])

            _, next_word = torch.max(prob, dim=-1)
            output.append(next_word)
            prev_y = self.trg_embed(next_word.unsqueeze(1))
            # attention_scores.append(self.decoder.attention.alphas.cpu().numpy())

        print(str(output))
        return torch.stack(output, dim=1)

    def to(self, device):
        super(Seq2SeqBiLSTM, self).to(device)
        self.device = device


class BiLSTMClassifier(nn.Module):

    def __init__(self, vocab_size: int, num_classes: int, input_dim=256, hidden_dim=512, num_layers=1, dropout=0.):
        super(BiLSTMClassifier, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, num_layers=num_layers, dropout=dropout)
        self.src_embed = nn.Embedding(vocab_size, input_dim)
        self.classification_head = AttentionClassificationHead(hidden_dim, num_classes)

    def forward(self, src, src_mask, src_lengths):
        encoder_hidden, encoder_final = self.encode(src, src_mask, src_lengths)
        return self.classification_head(encoder_hidden, src_mask)

    def encode(self, src, src_mask, src_lengths):
        return self.encoder(self.src_embed(src), src_mask, src_lengths)


def test_legacy():
    vocab_size = 50000
    model = LegacySeq2SeqBiLSTM(vocab_size=vocab_size, input_dim=128, hidden_dim=128)
    x = model(torch.randint(vocab_size, (4, 3)), torch.randint(vocab_size, (4, 10)))
    print(x.size())


def test_transformer():
    vocab_size = 50000
    model = SimpleTransformer(vocab_size)
    batch_shape = torch.Size((20, 8))

    i = torch.randint(vocab_size, batch_shape)
    l = torch.randint(vocab_size, batch_shape)
    i_masks = torch.ones(batch_shape)
    l_masks = torch.ones(batch_shape)

    y = model(i, l, i_masks, l_masks)
    print(y)
    print(y.size())


def test_LSTM_forward():
    vocab_size = 50000
    model = Seq2SeqBiLSTM(vocab_size)
    batch_shape = torch.Size((20, 8))

    i = torch.randint(vocab_size, batch_shape)
    l = torch.randint(vocab_size, batch_shape)
    i_masks = torch.ones(batch_shape)
    l_masks = torch.ones(batch_shape)
    src_lenghts = torch.full((batch_shape[0],), batch_shape[1])
    y = model(i, i_masks, l, l_masks, src_lenghts, src_lenghts)
    print(y)


def test_LSTM_generate():
    vocab_size = 50000
    model = Seq2SeqBiLSTM(vocab_size, device="cpu")
    batch_shape = torch.Size((2, 8))

    i = torch.randint(vocab_size, batch_shape)
    l = torch.randint(vocab_size, batch_shape)
    i_masks = torch.ones(batch_shape)
    l_masks = torch.ones(batch_shape)
    src_lenghts = torch.full((batch_shape[0],), batch_shape[1])

    y = model.generate(i, i_masks, src_lenghts, 9, 0, 1)
    print(y)
    print(y.size())


def test_LSTM_classifiy():
    vocab_size = 50000
    class_count = 5
    model = BiLSTMClassifier(vocab_size, class_count)
    batch_count = 20
    seq_len = 8
    batch_shape = torch.Size((batch_count, seq_len))

    i = torch.randint(vocab_size, batch_shape)
    i_masks = torch.ones(batch_shape)
    src_lenghts = torch.full((batch_count,), seq_len)
    y = model(i, i_masks, src_lenghts)
    print(y.size())
    assert y.size() == torch.Size((batch_count, class_count))

# test_LSTM_generate()
