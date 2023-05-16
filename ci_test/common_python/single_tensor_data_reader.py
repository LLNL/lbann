"""
Simple data reader that opens one file with one tensor. Used for unit testing.
"""
import numpy as np

# Lazy-load tensor
tensor = None


def lazy_load():
    # This file operates under the assumption that the working directory is set
    # to a specific experiment.
    global tensor
    if tensor is None:
        tensor = np.load('data.npy')
        assert len(tensor.shape) == 2


def get_sample(idx):
    lazy_load()
    return tensor[idx]


def num_samples():
    lazy_load()
    return tensor.shape[0]


def sample_dims():
    lazy_load()
    return (tensor.shape[1], )
