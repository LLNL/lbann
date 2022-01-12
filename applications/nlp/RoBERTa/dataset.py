import numpy as np

def get_sample(i):
    vals = range(513)
    return np.array(vals).flatten().astype(np.float64)

def num_samples():
    return 1

def sample_dims():
    return (513,)

