import numpy as np
import os


input_width = int(os.environ['INPUT_WIDTH'])
data_dir = os.environ['DATA_DIR']

assert input_width in [64,128, 256, 512]


w  = [input_width]*3 
w.insert(0,1)
dims = np.prod(w)
#Total sample is 101251 X 1 64 64 64
nsamples = 11000 #for 128^3
samples = None

# Sample access functions
def get_sample(index):
    global samples
    if samples is None:
      samples = np.load(data_dir, mmap_mode='r', allow_pickle=True)[:nsamples]
    return samples[index].flatten()

def num_samples():
    return nsamples

def sample_dims():
    return [dims]

