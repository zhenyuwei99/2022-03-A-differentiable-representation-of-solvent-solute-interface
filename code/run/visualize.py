#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : visualize.py
created time : 2022/04/02
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import os, shutil
import h5py
import torch
import numpy as np
from network import DATA_TYPE, DEVICE
from run import *

tcl_template = '''
mol new %s.pdb
mol delrep 0 top
mol selection \{protein\}
mol representation CPK 1.4 0.5 8 8
mol addrep top

mol new %s.xyz
mol delrep 0 top
mol representation Line
mol addrep top
'''

def generate_tcl_text(key):
    return tcl_template %(key, key)


if __name__ == '__main__':
    # Dir
    model_file = os.path.join(bak_dir, '06-less-layer', 'model.pt')
    out_dir = os.path.join(cur_dir, './out/image')
    pdb_dir = '/home/zhenyuwei/solvated-protein-data-set/solvated-protein-data-set/modified_data/test'
    # pdb_dir = '/home/zhenyuwei/hdd_data/solvated_protein_dataset/test'
    # Read data
    with h5py.File(test_dataset_file, 'r') as f:
            max_sequence_length = f['info/max_sequence_length'][()]
    # Initialization
    model = load_model(model_file, max_sequence_length)
    model.to(DEVICE)
    model.eval()
    # Dataset and dataloader
    dataset = SolvatedProteinDataset(test_dataset_file, is_return_key=True)
    # Predict
    for index in np.random.randint(0, len(dataset), 1):
        key = dataset[index][0]
        sequence = torch.tensor(dataset[index][1]).to(DATA_TYPE).to(DEVICE)
        sequence_coordinate, sequence_label = sequence[:, :3], sequence[:, 3].int()

        x, y, z = np.linspace(-60, 60, 150), np.linspace(-60, 60, 150), np.linspace(-60, 60, 150)
        X, Y, Z = np.meshgrid(x, y, z)
        coordinate = torch.tensor(np.stack([X.reshape([-1]), Y.reshape([-1]), Z.reshape([-1])]).T).to(DATA_TYPE).to(DEVICE)

        num_samples = coordinate.size()[0]
        result = np.zeros([num_samples])
        num_samples_per_epoch = 50000
        num_epochs = num_samples // num_samples_per_epoch

        with torch.no_grad():
            encoder_output = model.encoder(sequence_coordinate.unsqueeze(0), sequence_label.unsqueeze(0))
            for i in range(num_epochs):
                decoder_output = model.decoder(
                    coordinate[num_samples_per_epoch*i:num_samples_per_epoch*(i+1), :].unsqueeze(0), encoder_output
                )
                result[num_samples_per_epoch*i:num_samples_per_epoch*(i+1)] = model.ffn(decoder_output).squeeze().cpu()
                print('Finish %d:%d'%(num_samples_per_epoch*i, num_samples_per_epoch*(i+1)))
            decoder_output = model.decoder(
                coordinate[num_samples_per_epoch*num_epochs:, :].unsqueeze(0), encoder_output
            )
            result[num_samples_per_epoch*num_epochs:] = model.ffn(decoder_output).squeeze().cpu()
        result = coordinate[np.argwhere(result < 0.5), :].squeeze().cpu()
        # Visualize
        target_dir = os.path.join(out_dir, key)
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        os.mkdir(target_dir)
        xyz_file = os.path.join(target_dir, '%s.xyz' %key)
        with open(xyz_file, 'w') as f:
            print('%d\n' %result.size()[0], file=f)
            for coord in result:
                print('H\t%.5f\t%.5f\t%.5f' %(coord[0], coord[1], coord[2]), file=f)
        tcl_file = os.path.join(target_dir, 'vis.tcl')
        with open(tcl_file, 'w') as f:
            print(generate_tcl_text(key), file=f)
        os.system('cp %s/%s.pdb %s' %(pdb_dir, key, target_dir))