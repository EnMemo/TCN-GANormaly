import os, copy, torch
import numpy as np
from torch.utils.data import DataLoader

class DB(object):
    def __init__(self, features=[], seqlen=1):
        self.features = features
        self.seqlen = seqlen
    
    def __getitem__(self, index):
        return self.features[index:index+self.seqlen,:], self.features[index:index+self.seqlen,:]

    def __len__(self):
        return self.features.shape[0]-self.seqlen

    def get_dim(self):
        return self.features.shape[1]

    
    