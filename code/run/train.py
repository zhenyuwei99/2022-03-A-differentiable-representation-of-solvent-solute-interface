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
import h5py
import torch.optim as optim
import torch.utils.data as data
from network import NUM_FRAMES_PER_PROTEIN
from run import *

if __name__ == '__main__':
    # Read data
    with h5py.File(train_dataset_file, 'r') as f:
            max_sequence_length = f['info/max_sequence_length'][()]
    # Initialization
    if is_training_restart:
        with open(log_file, 'w') as f:
            print(network_info, file=f)
            print('# Start training at %s' %datetime.datetime.now().replace(microsecond=0), file=f)
            print(training_info, file=f)
        model = init_model(max_sequence_length)
        for m in model.modules():
            if isinstance(m, (torch.nn.Linear)):
                torch.nn.init.xavier_uniform_(m.weight)
    else:
        with open(log_file, 'a') as f:
            print('# Restart training at %s' %datetime.datetime.now().replace(microsecond=0), file=f)
            print(training_info, file=f)
        model = load_model(model_file, max_sequence_length)
    model.to(device)
    model.train()
    # Dataset and dataloader
    dataset = SolvatedProteinDataset(train_dataset_file)
    num_proteins = len(dataset)//NUM_FRAMES_PER_PROTEIN
    indices = (
        np.arange(0, num_proteins) * NUM_FRAMES_PER_PROTEIN +
        np.random.randint(0, NUM_FRAMES_PER_PROTEIN, size=(num_proteins))
    )
    np.random.shuffle(indices)
    sampler = data.SubsetRandomSampler(indices)
    loader = data.DataLoader(
        dataset, batch_size=batch_size, sampler=sampler,
        collate_fn=Collect(max_num_samples, data_type, device)
    )
    # Train
    criterion = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.99)
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
                        '- Epoch %02d, Iteration %06d, Loss %.6f' %(epoch+1, iteration+1, loss), file=f
                    )
            if iteration % save_interval == 0:
                save_model(model, model_file)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iteration += 1
        epoch += 1
