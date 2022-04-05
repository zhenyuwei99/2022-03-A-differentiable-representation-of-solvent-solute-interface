#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : predictor.py
created time : 2022/04/03
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import torch
import torch.nn as nn
import mdpy as md
from network.transformer import *

class Predictor:
    def __init__(self, model: Transformer) -> None:
        self.model = model

    def set_sequence(self, sequence):
        pass

    def set_grid(self, grid):
        pass

    def epsilon(self):
        pass

    def surface_area(self):
        pass