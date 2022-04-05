#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : test.py
created time : 2022/04/02
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import h5py
import torch
import torch.utils.data as data
from run import *

if __name__ == '__main__':
    # out_dir = '/home/zhenyuwei/nutstore/ZhenyuWei/Paper/2022-03-A-differentiable-representation-of-solvent-solute-interface/code/run/out/bak/03-larger-ffn'
    model_file = os.path.join(out_dir, 'model.pt')
    # Hyperparameter
    num_test_proteins = 50
    num_test_samples = 200
    # Read data
    with h5py.File(test_dataset_file, 'r') as f:
        max_sequence_length = f['info/max_sequence_length'][()]
    # Initialization
    model = load_model(model_file, max_sequence_length)
    model.to(device)
    model.eval()
    # Dataset and dataloader
    dataset = SolvatedProteinDataset(test_dataset_file)
    sampler = data.SubsetRandomSampler(
        np.random.randint(0, len(dataset), size=num_test_proteins)
    )
    loader = data.DataLoader(
        dataset, batch_size=1, sampler=sampler,
        collate_fn=Collect(num_test_samples, data_type, device)
    )
    num_total_samples, num_correct_samples = 0, 0
    with torch.no_grad():
        for sequence_coordinate, sequence_label, coordinate, label in loader:
            output = model(sequence_coordinate, sequence_label, coordinate)
            if batch_size == 1:
                output = output.unsqueeze(0)
            num_total_samples += output.numel()
            num_correct_samples += torch.count_nonzero(torch.abs(output - label) <= 0.50)
            print('Total Samples: %d, Accuracy: %.2f %%' %(num_total_samples, num_correct_samples/num_total_samples*100))