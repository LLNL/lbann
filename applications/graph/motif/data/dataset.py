import numpy as np

# Hard-coded options
_sample_size = 1234
_num_samples = 4321

# ----------------------------------------------
# Sample access functions
# ----------------------------------------------

def num_samples():
    return _num_samples

def sample_dims():
    return (_sample_size,)

def get_sample(index):
    ### @todo Real data. For best performance, return a NumPy array
    ### with dtype=np.float32.
    x = np.zeros(_sample_size, dtype=np.float32)
    x[index % _sample_size] = 1
    return x
