#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : data.py
created time : 2022/03/28
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import h5py
import numpy as np
import torch
import torch.utils.data as data
from utils import *

def sample_particles(num_samples: int, num_particles:int, num_protein_particles:int):
    res = []
    num_protein_samples = num_samples // 2
    num_solvent_samples = num_samples - num_protein_samples
    res.extend(list(np.random.randint(0, num_protein_particles, size=num_protein_samples)))
    res.extend(list(np.random.randint(num_protein_particles, num_particles, size=num_solvent_samples)))
    return res

class Collect:
    def __init__(self, pad_value):
        self.padding = pad_value

    def __call__(self, batch_data):
        '''
        Batch data:
        sequence: [sequence_length, 4]
        coordinate: [num_samples, 4]
        Return:
        sequence_coordinate: [batch_size, sequence_length, 3]
        sequence_label: [batch_size, sequence_length]
        coordinate: [batch_size, num_samples, 3]
        label: [batch_size, num_samples, 3]
        '''
        num_samples = np.random.randint(5, 50)
        max_sequence_length = 0
        sequence, coordinate_label = [], []
        for i, _ in batch_data:
            max_sequence_length = i.shape[0] if i.shape[0] > max_sequence_length else max_sequence_length
        for i, j in batch_data:
            sequence.append(np.vstack([
                i, np.zeros([max_sequence_length-i.shape[0], i.shape[1]])
            ]))
            coordinate_label.append(j[sample_particles(num_samples, j.shape[0], i.shape[0]), :])
        sequence = torch.tensor(np.stack(sequence), dtype=DATA_TYPE, device=DEVICE)
        coordinate_label = torch.tensor(np.stack(coordinate_label), dtype=DATA_TYPE, device=DEVICE)
        return sequence[:, :, :3], sequence[:, :, 3:].int(), coordinate_label[:, :, :3], coordinate_label[:, :, 3].int()


class SolvatedProteinDataset(data.Dataset):
    def __init__(self, dataset_file: str) -> None:
        super().__init__()
        self._dataset_file = dataset_file
        self._dataset = h5py.File(self._dataset_file, 'r')
        self._index_list = np.array(self._dataset['info/index_list'][()])
        self._num_particles = self._dataset['info/num_particles'][()]
        self._protein_list = self._dataset['info/protein_list'][()]
        self._num_proteins = self._dataset['info/num_proteins'][()]

    def __len__(self):
        return self._num_proteins

    def __getitem__(self, index: int):
        key = bytes.decode(self._protein_list[index])
        return (
            self._dataset['%s/sequence' %key][()],
            self._dataset['%s/coordinate_label' %key][()]
        )
    @property
    def num_particles(self):
        return self._num_particles