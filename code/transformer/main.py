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

import os
import datetime
import numpy as np
import h5py
import torch
import torch.utils.data as data
from dataset import *
from utils import *
from network import *

# Model Hyperparameters
dim_model = 64
dim_ffn = 256
dim_k = dim_v = 32
num_layers = 3
num_heads = 8
# Traning Hyperparameters
is_training_restart = True
num_epochs = 10
num_proteins_per_epoch = 5000
save_interval = 10
# Dirs
dataset_dir = '/home/zhenyuwei/Documents/solvated_protein_dataset'
cur_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(cur_dir, '../out/model/')
# File pathes
directory_file = os.path.join(out_dir, 'directory.txt')
dataset_file = os.path.join(dataset_dir, 'train.h5')
log_file = os.path.join(out_dir, 'train.log')
model_file = os.path.join(out_dir, 'model.pt')

def load_model(file_path: str):
    model = Transformer(
        dim_model=dim_model, dim_k=dim_k, dim_ffn=dim_ffn,
        directory_size=num_directory_keys,
        num_layers=num_layers, num_heads=num_heads,
        dropout=0.1, max_sequence_length=max_sequence_length + 100
    )
    model.load_state_dict(torch.load(file_path))
    return model

def save_model(model:nn.Module, file_path: str):
    torch.save(model.state_dict(), file_path)

if __name__ == '__main__':
    # Read data
    with h5py.File(dataset_file, 'r') as f:
        max_sequence_length = f['info/max_sequence_length'][()]
    directory, num_directory_keys = parse_directory(directory_file)
    # Initialization
    if is_training_restart:
        with open(log_file, 'w') as f:
            print('Start training at %s' %datetime.datetime.now().replace(microsecond=0), file=f)
        model = Transformer(
            dim_model=dim_model, dim_k=dim_k, dim_ffn=dim_ffn,
            directory_size=num_directory_keys,
            num_layers=num_layers, num_heads=num_heads,
            dropout=0.1, max_sequence_length=max_sequence_length+100
        )
    else:
        with open(log_file, 'a') as f:
            print('Restart training at %s' %datetime.datetime.now().replace(microsecond=0), file=f)
        model = load_model(model_file)
    model.train()
    # Dataset and dataloader
    dataset = SolvatedProteinDataset(dataset_file)
    sampler = data.SubsetRandomSampler(
        np.random.randint(0, len(dataset), size=num_proteins_per_epoch)
    )
    loader = data.DataLoader(dataset, batch_size=2, sampler=sampler, collate_fn=Collect(0))
    # Train
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99, weight_decay=0.001)
    iteration = 0
    for epoch in range(num_epochs):
        for sequence_coordinate, sequence_label, coordinate, label in loader:
            # output: [Batch_size, num_samples]
            output = model(sequence_coordinate, sequence_label, coordinate)
            loss = criterion(output.float(), label.float())
            with open(log_file, 'a') as f:
                print(
                    'Epoch %02d, Iteration %06d' %(epoch+1, iteration+1),
                    'loss =', '{:.6f}'.format(loss), file=f
                )
            if iteration % save_interval == 0:
                save_model(model, model_file)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iteration += 1
        epoch += 1