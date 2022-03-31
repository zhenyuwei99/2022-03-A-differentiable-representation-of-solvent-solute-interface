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

import time
import h5py
import numpy as np
import torch
import torch.utils.data as data

class Collect:
    def __init__(self, pad_value):
        self.padding = pad_value

    def __call__(self, batch_data):
        max_sequence_length = 0
        sequence, coordinate_label = [], []
        for i, _ in batch_data:
            max_sequence_length = i.shape[0] if i.shape[0] > max_sequence_length else max_sequence_length
        for i, j in batch_data:
            sequence.append(np.vstack([
                i, np.zeros([max_sequence_length-i.shape[0], i.shape[1]])
            ]))
            coordinate_label.append(j.tolist())
        sequence = torch.tensor(np.stack(sequence))
        coordinate_label = torch.tensor(np.stack(coordinate_label))
        return sequence[:, :, :3], sequence[:, :, 3:], coordinate_label[:, :3], coordinate_label[:, 3:]


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
        return self._num_particles

    def __getitem__(self, index: int):
        protein_index = np.searchsorted(self._index_list, index, side='right') - 1 # a[i-1] <= v < a[i]
        particle_index =  index - self._index_list[protein_index]
        key = bytes.decode(self._protein_list[protein_index])
        return (
            self._dataset['%s/sequence' %key][()],
            self._dataset['%s/coordinate_label' %key][(particle_index)]
        )
    @property
    def num_particles(self):
        return self._num_particles