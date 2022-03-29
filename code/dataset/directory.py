#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : directory.py
created time : 2022/03/29
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import os
from typing import IO
import mdpy as md

class CharmmTopologyParser:
    def __init__(self, *file_path_list) -> None:
        self._file_path_list = file_path_list
        self._target_block = ['RESI'] # ['RESI', 'PRES']
        self._directory = []
        self._index_directory = []

    def parse(self):
        for file_path in self._file_path_list:
            self._parse_single_file(file_path)
        self._generate_index_directory()

    def _generate_index_directory(self):
        self._index_directory = [i+1 for i, j in enumerate(self._directory)]

    def _parse_single_file(self, file_path: str):
        with open(file_path, 'r') as f:
            line, is_end = self._skip_block(f, f.readline())
            while not is_end:
                line = self._parse_block(f, line)
                line, is_end = self._skip_block(f, line)

    def _parse_block(self, f: IO, line:str):
        res_type, res_name = line.split()[:2]
        ter_particles = ['NH3', 'HC', 'CC', 'OC']
        if res_name == 'ALAD':
            return f.readline()
        while not 'BOND' in line:
            if line.startswith('ATOM'):
                self._directory.append('-'.join([
                    res_name, line.split()[2]
                ]))
            line = f.readline()
        # ter_particles = ['NH3', 'HC', 'CC', 'OC']
        # if res_type == 'RESI':
        #     for particle in ter_particles:
        #         self._directory.append('-'.join([
        #             res_name, particle
        #         ]))
        return f.readline()

    def _skip_block(self, f: IO, line: str):
        while line != '':
            for block in self._target_block:
                if line.startswith(block):
                    return line, False
            line = f.readline()
        return line, True

    def save(self, file_path: str):
        with open(file_path, 'w') as f:
            for i, j in zip(self._directory, self._index_directory):
                print('%s %i' %(i, j), file=f)

    @property
    def directory(self):
        return self._directory

    @property
    def index_directory(self):
        return self._index_directory

if __name__ == '__main__':
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.join(cur_dir, './data/'))
    out_dir = os.path.join(os.path.join(cur_dir, '../out/model/'))

    parser = CharmmTopologyParser(
        os.path.join(data_dir, 'top_all36_prot.rtf')
    )
    parser.parse()
    parser.save(os.path.join(out_dir, 'directory.txt'))
    print(len(parser.index_directory))