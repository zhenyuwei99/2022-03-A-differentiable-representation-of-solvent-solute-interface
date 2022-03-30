#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : main.py
created time : 2022/03/29
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import os
import multiprocessing as mp
from directory_creator import DirectoryCreator
from dataset_creator import DatasetCreator

if __name__ == '__main__':
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    is_create_directory = True
    is_create_test_dataset = not True
    is_create_train_dataset = True
    is_restart_test_dataset = not True
    is_restart_train_dataset = not True
    # Directory
    if is_create_directory:
        data_dir = os.path.join(os.path.join(cur_dir, './data/'))
        out_dir = os.path.join(os.path.join(cur_dir, '../out/model/'))
        directory_creator = DirectoryCreator(
            os.path.join(data_dir, 'top_all36_prot.rtf')
        )
        directory_creator.create_directory(os.path.join(out_dir, 'directory.txt'))
    # Dataset:
    out_dir = os.path.join(cur_dir, '../out/model')
    dataset_dir = '/home/zhenyuwei/hdd_data/solvated_protein_dataset'
    dataset_str_dir = os.path.join(dataset_dir, 'str')

    if is_create_test_dataset:
        # Test dataset
        dataset_test_dir = os.path.join(dataset_dir, 'test')
        creator = DatasetCreator(
            os.path.join(dataset_dir, 'test.h5'),
            os.path.join(out_dir, 'directory.txt'),
            os.path.join(dataset_dir, 'test.log'),
            dataset_test_dir,
            dataset_str_dir,
            is_restart=is_restart_test_dataset
        )
        creator.create_dataset(mp.cpu_count())
        creator.create_info_group()
    if is_create_train_dataset:
        # Training dataset
        dataset_train_dir = os.path.join(dataset_dir, 'train')
        creator = DatasetCreator(
            os.path.join(dataset_dir, 'train.h5'),
            os.path.join(out_dir, 'directory.txt'),
            os.path.join(dataset_dir, 'train.log'),
            dataset_train_dir,
            dataset_str_dir,
            is_restart=is_restart_train_dataset
        )
        creator.create_dataset(mp.cpu_count())
        creator.create_info_group()