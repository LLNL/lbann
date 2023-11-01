import numpy as np
import os


input_width = int(os.environ['INPUT_WIDTH'])
data_dir = os.environ['DATA_DIR']

assert input_width in [64,128, 256, 512]


w  = [input_width]*3 
w.insert(0,1)
dims = np.prod(w)
#Total sample is 101251 X 1 64 64 64
samples = np.load(data_dir, mmap_mode='r', allow_pickle=True)
nsamples = len(samples)

# Sample access functions
def get_sample(index):
    transpose_inds = [0] + np.random.permutation([1, 2, 3]).tolist()
    flip_axes = []
    for i in range(1, 4):
        if np.random.uniform() < 0.5:
            flip_axes.append(i)
    sample = np.flip(np.transpose(samples[index], transpose_inds), flip_axes)
    return sample.flatten()

def num_samples():
    return nsamples

def sample_dims():
    return [dims]

