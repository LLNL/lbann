from pathlib import Path 
from typing import List
import os
import numpy as np 


class MOFDataset():
    '''
    Custom Dataset loader for MOF data.  
    '''
    def __init__(self, path, transform=None):
        self.path = path
        path = Path(path)
        self.data = np.load(path)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.transform is not None:
            return self.transform(self.data[idx])
        else:
            return self.data[idx]
        

def test():
    data_dir = os.path.dirname(os.path.realpath(__file__))
    test_file_path = os.path.join(data_dir, 'mofs.npy')
    test_data = MOFDataset(test_file_path)
    
    print(test_data[0].shape)
    print(len(test_data))
    
if __name__ == '__main__':
    test()
