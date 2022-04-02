__author__ = "Zhenyu Wei"
__maintainer__ = "Zhenyu Wei"
__email__ = "zhenyuwei99@gmail.com"
__copyright__ = "Copyright 2021-2021, Southeast University and Zhenyu Wei"
__license__ = "MIT"

import torch
# Dataset constant
NUM_FRAMES_PER_PROTEIN = 25
# Network Constant
DEVICE = torch.device('cuda')
DATA_TYPE = torch.float32
DIM_COORDINATE = 3

import network.dataset as dataset
from network.transformer.dataset import SolvatedProteinDataset, Collect
from network.transformer.layer import Transformer

__all__ = [
    'SolvatedProteinDataset', 'Collect',
    'Transformer'
]