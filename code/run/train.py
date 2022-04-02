#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : train.py
created time : 2022/03/28
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import datetime
import numpy as np
import h5py
import torch
import torch.optim as optim
import torch.utils.data as data
from run import *

if __name__ == '__main__':
    # Read data
    with h5py.File(train_dataset_file, 'r') as f:
            max_sequence_length = f['info/max_sequence_length'][()]
    # Initialization
    if is_training_restart:
        with open(log_file, 'w') as f:
            print('Start training at %s' %datetime.datetime.now().replace(microsecond=0), file=f)
        model = init_model(max_sequence_length)
    else:
        with open(log_file, 'a') as f:
            print('Restart training at %s' %datetime.datetime.now().replace(microsecond=0), file=f)
        model = load_model(model_file, max_sequence_length)
    model.to(DEVICE)
    model.train()
    # Dataset and dataloader
    dataset = SolvatedProteinDataset(train_dataset_file)
    sampler = data.SubsetRandomSampler(
        np.random.randint(0, len(dataset), size=num_proteins_per_epoch)
    )
    loader = data.DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=Collect(max_num_samples))
    # Train
    criterion = nn.MSELoss()
    # criterion = nn.CrossEntropyLoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # optimizer = optim.SGD(model.parameters(), lr=1e-5, momentum=0.99, weight_decay=0.001)
    iteration = 0
    for epoch in range(num_epochs):
        for sequence_coordinate, sequence_label, coordinate, label in loader:
            # output: [Batch_size, num_samples]
            output = model(sequence_coordinate, sequence_label, coordinate)
            if batch_size == 1:
                output = output.unsqueeze(0)
            loss = criterion(output.float(), label.float())
            if iteration % log_interval == 0:
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
