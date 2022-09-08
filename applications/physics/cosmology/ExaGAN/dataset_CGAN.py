import numpy as np
import os

###Hack to get stuff working, should be replaced with more scalable hdf5 reader
input_width = int(os.environ['INPUT_WIDTH'])
data_dir = os.environ['DATA_DIR']

input_width = 128
assert input_width in [64,128, 256, 512]

#total sample in file: 12000
sample_size = 4000

data_dir = "/p/vast1/lbann/datasets/exagan/cGANc128/Om0.3_comb_Sg5n8n11_train.npy"

test_data_dir = "/p/vast1/lbann/datasets/exagan/cGANc128/interpol/Om0.3_Sg0.65_H70.0.npy"

w  = [input_width]*3 
w.insert(0,1)
dims = np.prod(w)
nsamples = sample_size*2
samples = None

# Sample access functions
def get_sample(index):
    global samples
    if samples is None:
      samples = np.load(data_dir, mmap_mode='r', allow_pickle=True)[:nsamples]
    f=samples[index].flatten()
    if index < sample_size : f = np.append(f,0.5)
    elif index >= sample_size and index < 2*sample_size : f = np.append(f,0.8)
    else: f= np.append(f,1.1) 
    return f

def num_samples():
    return nsamples

def sample_dims():
    return [dims+1]

ntest_samples=32
testsamples = None
##8800 total test exprolation samples
# Test sample access functions
def get_test_sample(index):
    global testsamples
    if testsamples is None:
      testsamples = np.load(test_data_dir, mmap_mode='r', allow_pickle=True)[:ntest_samples]
    f=testsamples[index].flatten()
    f = np.append(f,0.65) 
    return f

def num_test_samples():
    return ntest_samples

