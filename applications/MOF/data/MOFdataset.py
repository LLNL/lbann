import sys
import pickle
from pathlib import Path 
from typing import List
import os

import torch 
from torch.utils.data import Dataset 
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
from Voxel_MOF import Voxel_MOF



class MOFDataset(Dataset):
    '''
    Custom Dataset loader for MOF data.  
    '''
    def __init__(self, path, no_grid=False, no_loc=False,transform=None):
        self.path = path
        self.no_grid = no_grid
        self.no_loc = no_loc
        path = Path(path)
        # Voxel_MOF.test()
        with path.open("rb") as f:
            self.data = pickle.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.no_grid:
            return self.data[idx].loc_tensor.flatten()
        elif self.no_loc:
            return self.data[idx].grid_tensor    
        else:
            return self.data[idx].data

def test():
    data_dir = os.path.dirname(os.path.realpath(__file__))
    train_file_path = os.path.join(data_dir, 'data/mofs.p')  
    test_file_path = os.path.join(data_dir, 'data/mofs.p')

    training_data = MOFDataset(train_file_path)
    test_data = MOFDataset(test_file_path)

if __name__ == '__main__':
    test()
