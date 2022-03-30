#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : dataset_creator.py
created time : 2022/03/30
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

NUM_FRAMES_PER_PROTEIN = 25
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
        target_proteins = list(set([
            i.split('_')[0] for i in os.listdir(self._dataset_dir) \
                if i.endswith('.pdb') and not i.split('.pdb')[0] in finished_files
        ]))
        target_proteins.sort()
        target_proteins_list = np.array_split(target_proteins, num_processes)
        pool = mp.Pool(num_processes)
        manager = mp.Manager()
        dataset_lock = manager.Lock()
        log_lock = manager.Lock()
        for target_proteins in target_proteins_list:
            # self._parse_files(
            #         list(target_files), self._dataset_file, self._log_file,
            #         self._str_dir, self._directory, dataset_lock, log_lock
            #     )
            pool.apply_async(
                self._parse_files,
                args= (
                    list(target_proteins), self._dataset_file, self._log_file, finished_files,
                    self._dataset_dir, self._str_dir, self._directory, dataset_lock, log_lock
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
            lines = f.readlines()
            res = [
                line.split(' | ')[0].split('Info: ')[1] for line in lines if 'Info' in line
            ]
            # Failed job
            res.extend([
                line.split(' | ')[0].split('Warn: ')[1] for line in lines if 'Psf parse failed' in line
            ])
        return res

    @staticmethod
    def _parse_files(
        target_proteins: list[str], dataset_file: str, log_file: str, finished_files: list[str],
        dataset_dir: str, str_dir: str, directory: dict, dataset_lock, log_lock
    ):
        for protein in target_proteins:
            try:
                topology = md.io.PSFParser(os.path.join(str_dir, protein + '.psf')).topology
            except:
                log_lock.acquire()
                with open(log_file, 'a') as f:
                    for i in range(NUM_FRAMES_PER_PROTEIN):
                        file_info = '%s_%d' %(protein, i)
                        if not file_info in finished_files:
                            print('Warn: %s | Psf parse failed at %s' %(
                                file_info,  datetime.datetime.now().replace(microsecond=0)
                            ), file=f)
                log_lock.release()
                continue
            for i in range(NUM_FRAMES_PER_PROTEIN):
                file_path = os.path.join(dataset_dir, '%s_%d.pdb' %(protein, i))
                file_info = '%s_%d' %(protein, i)
                if not file_info in finished_files:
                    # Read data
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
                    dataset_lock.acquire()
                    with h5.File(dataset_file, 'a') as f:
                        if file_info in f.keys():
                            f.__delitem__(file_info)
                        f['%s/sequence' %file_info] = sequence_data
                        f['%s/coordinate_label' %file_info] = coordinate_label_data
                        f['%s/num_protein_particles' %file_info] = sequence_data.shape[0] # Ignore unrecognized protein particles
                        f['%s/num_particles' %file_info] = coordinate_label_data.shape[0]
                    dataset_lock.release()
                    log_lock.acquire()
                    with open(log_file, 'a') as f:
                        print('Info: %s | Finish at %s' %(file_info, datetime.datetime.now().replace(microsecond=0)), file=f)
                    log_lock.release()