import numpy as np
import h5py
from torch.utils.data import Dataset
from glob import glob
import os
import torch.distributed as dist
from tqdm import tqdm


class CosmoflowDataset(Dataset):
    def __init__(self, data_dir, preload=False):
        self.preload = preload

        self.files = glob(os.path.join(data_dir, "*.hdf5"))[
            dist.get_rank() :: dist.get_world_size()
        ]

        if self.preload:
            self.x = []
            self.y = []
            for file in tqdm(self.files, desc="Preloading Data"):
                with h5py.File(file) as f:
                    self.x.append(np.array(f["full"]))
                    self.y.append(np.array(f["unitPar"]))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        if self.preload:
            return self.x[index], self.y[index]

        with h5py.File(self.files[index]) as f:
            x = np.array(f["full"])
            y = np.array(f["unitPar"])

        return x, y
