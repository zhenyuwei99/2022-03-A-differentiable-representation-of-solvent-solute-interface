#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : dataset.py
created time : 2022/03/29
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import os
import datetime
import mdpy as md
import numpy as np
import h5py as h5
import multiprocessing as mp

class DatasetCreator:
    def __init__(self, dataset_file: str, directory_file: str, log_file:str, dataset_dir: str, str_dir: str, is_restart=False) -> None:
        # Read input
        if not dataset_file.endswith('.h5'):
            raise KeyError('File path should endwith .h5')
        self._dataset_file = dataset_file
        self._directory_file = directory_file
        self._dataset_dir = dataset_dir
        self._str_dir = str_dir
        self._log_file = log_file
        self._is_restart = is_restart

        # Attributes
        self._directory = {}
        self._directory_keys = []
        self._parse_directory_file()
        # Refresh file
        if is_restart:
            open(self._log_file, 'w').close()
            h5.File(self._dataset_file, 'w')

    def create_dataset(self, num_processes: int=10):
        finished_files = self._parse_log_file()
        target_files = [
            os.path.join(self._dataset_dir, i) for i in os.listdir(self._dataset_dir) \
                if i.endswith('.pdb') and not i.split('.pdb')[0] in finished_files
        ]
        target_files.sort()
        target_files_list = np.array_split(target_files, num_processes)
        pool = mp.Pool(num_processes)
        manager = mp.Manager()
        dataset_lock = manager.Lock()
        log_lock = manager.Lock()
        for target_files in target_files_list:
            pool.apply_async(
                self._parse_files,
                args= (
                    list(target_files), self._dataset_file, self._log_file,
                    self._str_dir, self._directory, dataset_lock, log_lock
                )
            )
        pool.close()
        pool.join()

    def _parse_directory_file(self):
        with open(self._directory_file, 'r') as f:
            line = f.readline()
            while line != '':
                info = line.split()
                self._directory[info[0]] = int(info[1])
                line = f.readline()
        self._directory_keys = self._directory.keys()

    def _parse_log_file(self):
        with open(self._log_file, 'r') as f:
            res = [line.split(' | ')[0].split('Info: ')[1] for line in f.readlines() if 'Info' in line]
        return res

    @staticmethod
    def _parse_files(
        file_path_list: str, dataset_file: str, log_file: str,
        str_dir: str, directory: dict, dataset_lock, log_lock
    ):
        for file_path in file_path_list:
            if not file_path.endswith('.pdb'):
                raise KeyError('File path should endwith .pdb')
            file_info = file_path.split('/')[-1].split('.pdb')[0]
            file_name = file_info.split('_')[0]
            # Read data
            topology = md.io.PSFParser(os.path.join(str_dir, file_name + '.psf')).topology
            positions = md.io.PDBParser(file_path).positions
            # Parse data
            protein_matrix_ids = md.utils.select(topology, [{'protein': []}])
            num_proteins = len(protein_matrix_ids)

            sequence_data = []
            directory_keys = directory.keys()
            for matrix_id in protein_matrix_ids:
                particle = topology.particles[matrix_id]
                particle_key = '%s-%s' %(
                    particle.molecule_type, particle.particle_type
                )
                if particle_key in directory_keys:
                    sequence_data.append(np.array(
                        list(positions[matrix_id, :]) + [directory[particle_key]]
                    ))
                else:
                    log_lock.acquire()
                    with open(log_file, 'a') as f:
                        print('Warn: %s | Failed at %s-%s, particle id %d' %(
                            file_info, particle.molecule_type,
                            particle.particle_type, particle.particle_id
                        ), file=f)
                    log_lock.release()
            sequence_data = np.stack(sequence_data)
            coordinate_label_data = np.zeros([positions.shape[0], 4])
            coordinate_label_data[:, :3] = positions[:, :]
            coordinate_label_data[:num_proteins, 3] = 0
            coordinate_label_data[num_proteins:, 3] = 1

            with h5.File(dataset_file, 'a') as f:
                dataset_lock.acquire()
                f['%s/sequence' %file_info] = sequence_data
                f['%s/coordinate_label' %file_info] = coordinate_label_data
                f['%s/num_protein_particles' %file_info] = sequence_data.shape[0] # Ignore unrecognized protein particles
                f['%s/num_particles' %file_info] = coordinate_label_data.shape[0]
                dataset_lock.release()
            with open(log_file, 'a') as f:
                log_lock.acquire()
                print('Info: %s | Finish at %s' %(file_info, datetime.datetime.now().replace(microsecond=0)), file=f)
                log_lock.release()
