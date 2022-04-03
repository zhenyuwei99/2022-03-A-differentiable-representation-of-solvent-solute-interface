#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : utils.py
created time : 2022/03/31
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import torch
import torch.nn as nn

# Parse directory file
def parse_directory(directory_file: str):
    directory, num_directory_keys = [], 0
    with open(directory_file, 'r') as f:
        for line in f.readlines():
            if '-' in line:
                directory.append(line.split()[0])
                num_directory_keys += 1
    return directory, num_directory_keys

# Position encoding
class PositionEncoding(nn.Module):
    def __init__(
        self, dim_model, dropout, max_length,
        data_type, device
    ) -> None:
        super().__init__()
        # Input
        self._data_type = data_type
        self._device = device
        # Layer
        self._dropout = nn.Dropout(p=dropout)
        self._encoding = torch.zeros(max_length, dim_model).to(self._data_type).to(self._device)
        # Do not upgrade during optimization
        self._encoding.requires_grad = False

        pos = torch.arange(0, max_length).to(self._data_type).to(self._device)
        pos = pos.float().unsqueeze(dim=1) # [max_length x 1]
        if dim_model % 2 != 0:
            nominator = torch.arange(0, dim_model+1, step=2).to(self._data_type).to(self._device)
            self._encoding[:, 0::2] = torch.sin(pos / (10000 ** (nominator / dim_model)))
            nominator = torch.arange(0, dim_model-1, step=2).to(self._data_type).to(self._device)
            self._encoding[:, 1::2] = torch.cos(pos / (10000 ** (nominator / dim_model)))
        else:
            nominator = torch.arange(0, dim_model, step=2).to(self._data_type).to(self._device)
            self._encoding[:, 0::2] = torch.sin(pos / (10000 ** (nominator / dim_model)))
            self._encoding[:, 1::2] = torch.cos(pos / (10000 ** (nominator / dim_model)))

    def forward(self, x: torch.Tensor):
        '''
        x: [batch_size, sequence_length, dim_model]
        '''
        output = x + self._encoding[:x.size(1)]
        output.masked_fill_(x==0, 0)
        return self._dropout(output)

# Padding mask

# ScaledDotProductAttention
class ScaledDotProductAttention(nn.Module):
    def __init__(self, data_type, device) -> None:
        super().__init__()
        self._data_type = data_type
        self._device = device

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
        '''
        Q: [batch_size, num_heads, len_q, dim_q]
        K: [batch_size, num_heads, len_k, dim_k]
        V: [batch_size, num_heads, len_v, dim_v]

        Note:
        - dim_q == dim_k
        - len_k == len_v
        - num_heads * dim_v = d_model
        '''
        # scores : [batch_size, num_heads, len_q, len_k]
        dim_k = torch.tensor(K.size()[-1]).to(self._data_type)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / torch.sqrt(dim_k)
        # attention : [batch_size, num_heads, len_q, len_k]
        attention = nn.Softmax(dim=-1)(scores)
        # context: [batch_size, num_heads, len_q, dim_v]
        context = torch.matmul(attention, V)
        return context

class MultiHeadAttention(nn.Module):
    def __init__(self, dim_model, dim_k, num_heads, dropout, data_type, device) -> None:
        super().__init__()
        # Input
        if dim_model % num_heads != 0:
            raise KeyError('dim_model is not proportional to the num_heads')
        self._dim_model = dim_model
        self._dim_k = dim_k
        self._dim_v = self._dim_model // num_heads
        self._num_heads = num_heads
        self._dropout = dropout
        self._data_type = data_type
        self._device = device
        # Layer
        self.W_Q = nn.Linear(
            self._dim_model, self._dim_k * self._num_heads, bias=False
        ).to(self._data_type).to(self._device)
        self.W_K = nn.Linear(
            self._dim_model, self._dim_k * self._num_heads, bias=False
        ).to(self._data_type).to(self._device)
        self.W_V = nn.Linear(
            self._dim_model, self._dim_v * self._num_heads, bias=False
        ).to(self._data_type).to(self._device)
        self.layer_norm = nn.LayerNorm(self._dim_model).to(self._data_type).to(self._device)
        self.scaled_dot_product_attention = ScaledDotProductAttention(self._data_type, self._device)
        self.dropout = nn.Dropout(p=self._dropout).to(self._data_type).to(self._device)

    def forward(self, input_Q: torch.Tensor, input_K: torch.Tensor, input_V: torch.Tensor):
        '''
        input_Q: [batch_size, len_q, dim_model]
        input_K: [batch_size, len_k, dim_model]
        input_V: [batch_size, len_v, dim_model]

        Note:
        - len_k == len_v as it is always generate from the same source
        - len_k may be difference with len_q in decoder layer
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # Q: [batch_size, n_heads, len_q, dim_q == dim_k]
        Q = self.W_Q(input_Q).view(batch_size, -1, self._num_heads, self._dim_k).transpose(1,2)
        # K: [batch_size, n_heads, len_k, dim_k]
        K = self.W_K(input_K).view(batch_size, -1, self._num_heads, self._dim_k).transpose(1,2)
        # V: [batch_size, n_heads, len_v, dim_v]
        V = self.W_V(input_V).view(batch_size, -1, self._num_heads, self._dim_v).transpose(1,2)

        # context: [batch_size, n_heads, sequence_length, dim_v]
        context = self.scaled_dot_product_attention(Q, K, V)
        # context: [batch_size, len_q, n_heads * dim_v = dim_model]
        context = context.transpose(1, 2).reshape(batch_size, -1, self._num_heads * self._dim_v)
        # Dropout before add
        context = self.dropout(context)
        # Dropout after add
        return self.dropout(self.layer_norm(context + residual))

# Position-wise fully connected feed-forward network
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, dim_model, dim_ff, data_type, device) -> None:
        super().__init__()
        # Input
        self._dim_model = dim_model
        self._dim_ff = dim_ff
        self._data_type = data_type
        self._device = device
        # Layer
        self.ffn = nn.Sequential(
            nn.Linear(self._dim_model, self._dim_ff, bias=False).to(self._data_type),
            nn.ReLU(),
            nn.Linear(self._dim_ff, self._dim_model, bias=False).to(self._data_type)
        ).to(self._device)
        self.layer_norm = nn.LayerNorm(self._dim_model).to(self._data_type).to(self._device)

    def forward(self, inputs: torch.Tensor):
        '''
        inputs: [batch_size, sequence_length, dim_model]
        '''
        residual = inputs
        output = self.ffn(inputs)
        # [batch_size, sequence_length, dim_model]
        return self.layer_norm(output + residual)
