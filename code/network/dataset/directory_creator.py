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

from typing import IO

class DirectoryCreator:
    def __init__(self, *file_path_list) -> None:
        self._file_path_list = file_path_list
        self._target_block = ['RESI', 'PRES']
        self._directory = []
        self._index_directory = []
        self._residue_list = []
        self.parse()

    def parse(self):
        for file_path in self._file_path_list:
            self._parse_single_file(file_path)
        self._generate_index_directory()

    def _generate_index_directory(self):
        self._directory = list(set(self._directory))
        self._directory.sort()
        self._index_directory = [i+1 for i, j in enumerate(self._directory)]

    def _parse_single_file(self, file_path: str):
        with open(file_path, 'r') as f:
            line, is_end = self._skip_block(f, f.readline(), 'RESI')
            while not is_end:
                line = self._parse_RESI_block(f, line)
                line, is_end = self._skip_block(f, line, 'RESI')

        with open(file_path, 'r') as f:
            line, is_end = self._skip_block(f, f.readline(), 'PRES')
            while not is_end:
                line = self._parse_PRES_block(f, line)
                line, is_end = self._skip_block(f, line, 'PRES')

    def _parse_RESI_block(self, f: IO, line:str):
        res_name = line.split()[1]
        if res_name != 'ALAD':
            self._residue_list.append(res_name)
        while not 'BOND' in line:
            if res_name != 'ALAD' and line.startswith('ATOM'):
                self._directory.append('-'.join([
                    res_name, line.split()[2]
                ]))
            line = f.readline()
        return f.readline()

    def _parse_PRES_block(self, f: IO, line:str):
        res_name = line.split()[1]
        if 'C-terminus' in line or 'N-terminus' in line or 'N-Terminal' in line:
            if not 'dipeptide' in line: # Remove dipeptide info
                is_specified_patch = False
                for residue in self._residue_list:
                    if residue in res_name and not 'dipeptide' in line: # Reiddue specified patching
                        is_specified_patch = residue
                if is_specified_patch != False:
                    while not 'BOND' in line:
                        if line.startswith('ATOM'):
                            self._directory.append('-'.join([
                                is_specified_patch, line.split()[2]
                            ]))
                        line = f.readline()
                else:
                    atoms = []
                    while not 'BOND' in line:
                        if line.startswith('ATOM'):
                            atoms.append(line.split()[2])
                        line = f.readline()
                    for residue in self._residue_list:
                        for atom in atoms:
                            self._directory.append('-'.join([
                                residue, atom
                            ]))
        else: # No N, C terminus but residue specified
            is_specified_patch = False
            for residue in self._residue_list:
                if residue in line and not 'dipeptide' in line: # Reiddue specified patching
                    is_specified_patch = residue
            if is_specified_patch != False:
                while not 'BOND' in line:
                    if line.startswith('ATOM'):
                        self._directory.append('-'.join([
                            is_specified_patch, line.split()[2]
                        ]))
                    line = f.readline()
        return f.readline()

    def _skip_block(self, f: IO, line: str, target_block: str):
        while line != '':
            if line.startswith(target_block):
                return line, False
            line = f.readline()
        return line, True

    def create_directory(self, file_path: str):
        with open(file_path, 'w') as f:
            for i, j in zip(self._directory, self._index_directory):
                print('%s %i' %(i, j), file=f)

    @property
    def directory(self):
        return self._directory

    @property
    def index_directory(self):
        return self._index_directory