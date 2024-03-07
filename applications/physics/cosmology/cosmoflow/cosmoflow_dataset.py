import numpy as np
from glob import glob
from lbann.util.data import Sample, SampleDims, Dataset, DistConvDataset
import h5py as h5
import os

    
class CosmoFlowDataset(DistConvDataset):
    def __init__(self, data_dir, input_width, num_secrets):
        self.data_dir = data_dir
        self.input_width = input_width
        self.num_secrets = num_secrets
        self.samples = glob(os.path.join(data_dir, '*.hdf5'))
        self.samples.sort()
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index) -> Sample:
        data = h5.File(self.samples[index], 'r')
        slice_width = self.input_width // self.num_io_partitions
        slice_ind = self.rank % self.num_io_partitions
        full = data['full'][:,
                            slice_ind*slice_width:(slice_ind+1)*slice_width,
                            :self.input_width,
                            :self.input_width].astype(np.float32)
        par = data['unitPar'][:].astype(np.float32)
        return Sample(sample=np.ascontiguousarray(full), response=par)
    
    def get_sample_dims(self):
        return SampleDims(sample=[4, self.input_width, self.input_width, self.input_width], response=self.num_secrets)
