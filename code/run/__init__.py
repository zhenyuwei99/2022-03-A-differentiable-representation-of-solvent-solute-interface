__author__ = "Zhenyu Wei"
__maintainer__ = "Zhenyu Wei"
__email__ = "zhenyuwei99@gmail.com"
__copyright__ = "Copyright 2021-2021, Southeast University and Zhenyu Wei"
__license__ = "MIT"


import os
import torch
from run import *
from network.transformer.dataset import *
from network.transformer.utils import *
from network.transformer.layer import *
torch.cuda.set_per_process_memory_fraction(1., 0)

# Dir Hyperparameters
dataset_dir = '/home/zhenyuwei/Documents/solvated_protein_dataset'
# Model Hyperparameters
dim_model = 32
dim_ffn = 256
dim_k = dim_v = 32
num_layers = 6
num_heads = 4
# Traning Hyperparameters
is_training_restart = not True
batch_size = 1
max_num_samples = 750
num_epochs = 500
num_proteins_per_epoch = 5000
learning_rate = 1e-5
dropout_prob = 0.1
weight_decay = 0
save_interval = 100
log_interval = 50

# Other dirs
cur_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(cur_dir, './out/model/')
# File pathes
directory_file = os.path.join(out_dir, 'directory.txt')
train_dataset_file = os.path.join(dataset_dir, 'train.h5')
test_dataset_file = os.path.join(dataset_dir, 'test.h5')
log_file = os.path.join(out_dir, 'train.log')
model_file = os.path.join(out_dir, 'model.pt')
# Parse directory
directory, directory_size = parse_directory(directory_file)

# Model saving and loading
def init_model(max_sequence_length: int):
    model = Transformer(
        dim_model=dim_model, dim_k=dim_k, dim_ffn=dim_ffn,
        directory_size=directory_size,
        num_layers=num_layers, num_heads=num_heads,
        dropout=dropout_prob, max_sequence_length=max_sequence_length + 100
    )
    return model

def load_model(file_path: str, max_sequence_length: int):
    model = init_model(max_sequence_length)
    model.load_state_dict(torch.load(file_path))
    return model

def save_model(model:nn.Module, file_path: str):
    torch.save(model.state_dict(), file_path)