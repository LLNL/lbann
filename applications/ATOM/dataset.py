import os
import numpy as np
import json

#@todo, get rid of json and pass all variable here
# the idea here is to use the same code with abritrary sets of data
with open(os.environ['DATA_CONFIG'], 'rb') as handle:
    config = json.load(handle)

pad_index = config['pad_index']
max_seq_len = int(os.environ['MAX_SEQ_LEN']) 
samples = np.load(os.environ['DATA_PATH'], allow_pickle=True)


# Sample access functions
def get_sample(index):
    sample = samples[index]
    if len(sample) < max_seq_len:
        sample = np.concatenate((sample, np.full(max_seq_len-len(sample), pad_index)))
    else:
        sample = np.resize(sample, max_seq_len)
    return sample

def num_samples():
    return samples.shape[0]

def sample_dims():
    return [max_seq_len]

