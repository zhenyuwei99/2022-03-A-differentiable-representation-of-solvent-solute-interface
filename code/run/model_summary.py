#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : model_summary.py
created time : 2022/04/03
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

from run import *

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    # Read data
    with h5py.File(train_dataset_file, 'r') as f:
        max_sequence_length = f['info/max_sequence_length'][()]
    # Initialization
    model_file = os.path.join(bak_dir, '06-larger-batch-size-less-layer', 'model.pt')
    model = load_model(model_file, max_sequence_length=max_sequence_length)
    print('parameters_count:',count_parameters(model))