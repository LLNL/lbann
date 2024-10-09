import numpy as np
import os

data_file = '/p/vast1/lbann/datasets/HRRL/images_spectra_val_set.npy'

sample_size = 90201 #size of one sample, flattened (For PROBIES, 300x300 image + 201 spectra array)
nsamples = 631 

samples = None

# Sample access functions
def get_sample(index):
    global samples
    if samples is None:
      samples_raw = np.load(data_file, mmap_mode='r', allow_pickle=True)
      samples = np.reshape(samples_raw,(nsamples,sample_size))  
    return samples[index]

def num_samples():
    return nsamples

def sample_dims():
    return [sample_size]