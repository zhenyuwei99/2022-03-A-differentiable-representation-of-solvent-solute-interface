#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : main.py
created time : 2022/03/28
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import os, time
import numpy as np
import h5py
import torch
import torch.utils.data as data
from dataset import SolvatedProteinDataset, Collect
from utils import *
from network import *

if __name__ == '__main__':
    # Dirs and pathes
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(cur_dir, '../out/model/')
    directory_file = os.path.join(out_dir, 'directory.txt')
    dataset_file = '/home/zhenyuwei/Documents/solvated_protein_dataset/train.h5'
    # Hyper Parameters
    dim_model = 64
    dim_ffn = 256
    dim_k = dim_v = 32
    num_layers = 3
    num_heads = 8
    # Read data
    with h5py.File(dataset_file, 'r') as f:
        max_sequence_length = f['info/max_sequence_length'][()]
    directory, num_directory_keys = parse_directory(directory_file)
    dataset = SolvatedProteinDataset(dataset_file)

    num_samples_per_epoch = 100000
    # Train
    num_epochs = 10
    model = Transformer(
        dim_model=dim_model, dim_k=dim_k, dim_ffn=dim_ffn,
        directory_size=num_directory_keys,
        num_layers=num_layers, num_heads=num_heads,
        dropout=0.1, max_sequence_length=max_sequence_length + 100
    )
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)
    sampler = data.SubsetRandomSampler(
        np.random.randint(0, dataset.num_particles, size=num_samples_per_epoch)
    )
    loader = data.DataLoader(dataset, batch_size=2, sampler=sampler, collate_fn=Collect(0))
    for epoch in range(num_epochs):
        for sequence_coordinate, sequence_label, coordinate, label in loader:
            # output: [Batch_size]
            output = model(sequence_coordinate, sequence_label, coordinate)
            loss = criterion(output.float(), label.float())
            print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
            print(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()