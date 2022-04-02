#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : layer.py
created time : 2022/03/30
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import torch
import torch.nn as nn
from network import DATA_TYPE, DEVICE, DIM_COORDINATE
from network.transformer.utils import *

# Encoder Layer
class EncoderLayer(nn.Module):
    def __init__(self, dim_model, dim_k, dim_ff, num_heads) -> None:
        super().__init__()
        self._dim_model = dim_model
        self._dim_k = dim_k
        self._dim_ff = dim_ff
        self._num_heads = num_heads
        self._encoder_self_attention = MultiHeadAttention(
            self._dim_model, self._dim_k, self._num_heads
        )
        self._poswise_ffn = PoswiseFeedForwardNet(
            self._dim_model, self._dim_ff
        )

    def forward(self, encoder_input: torch.Tensor):
        '''
        encoder_input: [batch_size, sequence_length, dim_model]
        '''
        # encoder_output: [batch_size, sequence_length, dim_model]
        encoder_output = self._encoder_self_attention(
            encoder_input, encoder_input, encoder_input
        )
        encoder_output = self._poswise_ffn(encoder_output)
        return encoder_output

# Encoder
class Encoder(nn.Module):
    def __init__(
        self, dim_model, dim_k, dim_ffn,
        directory_size, num_layers, num_heads,
        dropout=0.1, max_sequence_length=10000
    ) -> None:
        super().__init__()
        self._dim_model = dim_model
        self._dim_model_embeding = self._dim_model - DIM_COORDINATE
        self._dim_k = dim_k
        self._dim_ffn = dim_ffn
        self._directory_size = directory_size
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._dropout = dropout
        self._max_sequence_length = max_sequence_length
        self.sequence_embeding = nn.Embedding(
            self._directory_size, self._dim_model_embeding, padding_idx=0
        ).to(DEVICE)
        self.position_encoding = PositionEncoding(
            self._dim_model_embeding, dropout=self._dropout,
            max_length=self._max_sequence_length
        )
        self.layers = nn.ModuleList([
            EncoderLayer(
                self._dim_model, self._dim_k, self._dim_ffn, self._num_heads
            ) for _ in range(self._num_layers)
        ])

    def forward(self, sequence_coordinate: torch.Tensor, sequence_label: torch.Tensor):
        '''
        sequence_coordinate: [batch_size, sequence_length, DIM_COODINATE]
        sequence_label: [batch_size, sequence_length]
        '''
        # sequence_label [batch_size, sequence_length, dim_model_embeding]
        sequence_label = self.sequence_embeding(sequence_label).squeeze(2)
        sequence_label = self.position_encoding(sequence_label)
        # encoder_input [batch_size, sequence_length, dim_model]
        encoder_output = torch.cat([sequence_coordinate, sequence_label], dim=2)
        for layer in self.layers:
            # encoder_output: [batch_size, sequence_length, d_model]
            encoder_output = layer(encoder_output)
        return encoder_output

# Decoder Layer
class DecoderLayer(nn.Module):
    def __init__(self, dim_model, dim_k, dim_ff, num_heads) -> None:
        super().__init__()
        self._dim_model = dim_model
        self._dim_k = dim_k
        self._dim_ff = dim_ff
        self._num_heads = num_heads
        self._encoder_self_attention = MultiHeadAttention(
            self._dim_model, self._dim_k, self._num_heads
        )
        self._poswise_ffn = PoswiseFeedForwardNet(
            self._dim_model, self._dim_ff
        )

    def forward(self, decoder_input: torch.Tensor, encoder_input: torch.Tensor):
        '''
        encoder_input: [batch_size, sequence_length, dim_model]
        decoder_input: [batch_size, num_samples, dim_model]
        '''
        # decoder_output: [batch_size, num_samples, dim_model]
        decoder_output = self._encoder_self_attention(
            decoder_input, encoder_input, encoder_input
        )
        decoder_output = self._poswise_ffn(decoder_output)
        return decoder_output

# Decoder
class Decoder(nn.Module):
    def __init__(self, dim_model, dim_k, dim_ffn, directory_size, num_layers, num_heads) -> None:
        super().__init__()
        self._dim_model = dim_model
        self._dim_k = dim_k
        self._dim_ffn = dim_ffn
        self._directory_size = directory_size
        self._num_layers = num_layers
        self._num_heads = num_heads
        self.W_Input = nn.Linear(DIM_COORDINATE, self._dim_model, bias=False).to(DATA_TYPE).to(DEVICE)
        self.layers = nn.ModuleList([
            DecoderLayer(
                self._dim_model, self._dim_k, self._dim_ffn, self._num_heads
            ) for _ in range(self._num_layers)
        ])

    def forward(self, decoder_input: torch.Tensor, encoder_input: torch.Tensor):
        '''
        decoder_input: [batch_size, num_samples, 3]
        encoder_input: [batch_size, sequence_length, dim_model]
        '''
        # decoder_input: [batch_size, num_samples, dim_model]
        decoder_output = self.W_Input(decoder_input)
        for layer in self.layers:
            # decoder: [batch_size, num_samples, dim_model]
            decoder_output = layer(decoder_output, encoder_input)
        return decoder_output

# Transformer
class Transformer(nn.Module):
    def __init__(
        self, dim_model, dim_k, dim_ffn,
        directory_size, num_layers, num_heads,
        dropout=0.1, max_sequence_length=10000
    ) -> None:
        super().__init__()
        self._dim_model = dim_model
        self._dim_model_embeding = self._dim_model - DIM_COORDINATE
        self._dim_k = dim_k
        self._dim_ffn = dim_ffn
        self._directory_size = directory_size
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._dropout = dropout
        self._max_sequence_length = max_sequence_length
        self.encoder = Encoder(
            dim_model=self._dim_model,
            dim_k=self._dim_k,
            dim_ffn=self._dim_ffn,
            directory_size=self._directory_size,
            num_layers=self._num_layers,
            num_heads=self._num_heads,
            dropout=self._dropout,
            max_sequence_length=self._max_sequence_length
        )
        self.decoder = Decoder(
            dim_model=self._dim_model,
            dim_k=self._dim_k,
            dim_ffn=self._dim_ffn,
            directory_size=self._directory_size,
            num_layers=self._num_layers,
            num_heads=self._num_heads
        )
        self.ffn = nn.Sequential(
            nn.Linear(dim_model, self._dim_ffn).to(DATA_TYPE),
            nn.ReLU(),
            nn.Linear(self._dim_ffn, 1).to(DATA_TYPE),
            nn.Sigmoid()
        ).to(DEVICE)

    def forward(
        self, sequence_coordinate: torch.Tensor,
        sequence_label: torch.Tensor, coordinate: torch.Tensor
    ):
        '''
        sequence_coordinate: [batch_size, sequence_length, DIM_COODINATE]
        sequence_label: [batch_size, sequence_length]
        coordinate: [batch_size, 3]
        '''
        # encoder_output: [batch_size, sequence, dim_model]
        encoder_output = self.encoder(sequence_coordinate, sequence_label)
        # decoder_output: [batch_size, 1, dim_model]
        decoder_output = self.decoder(coordinate, encoder_output)
        # output: [batch_size]
        output = self.ffn(decoder_output).squeeze()
        return output