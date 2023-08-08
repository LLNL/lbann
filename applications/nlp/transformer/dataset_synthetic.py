"""
Synthetic dataset for benchmarking the transformer sample
"""
import numpy as np

# ----------------------------------------------
# Options
# ----------------------------------------------

# Note: Sequence lengths for WMT 2014 have mean 29.05, standard
# deviation 16.20, and max 484.
sequence_length = 64
_vocab_size = 32000

# ----------------------------------------------
# Setup
# ----------------------------------------------

pad_index = 0

# ----------------------------------------------
# Sample access functions
# ----------------------------------------------


def get_train_sample(index):
    return np.random.randint(0,
                             _vocab_size,
                             size=(2 * sequence_length, ),
                             dtype=np.int32)


def get_val_sample(index):
    sample_one = np.random.randint(0,
                                   _vocab_size,
                                   size=(sequence_length, ),
                                   dtype=np.int32)
    sample_two = np.random.randint(0,
                                   _vocab_size,
                                   size=(sequence_length, ),
                                   dtype=np.int32)
    return sample_one, sample_two


def num_train_samples():
    return 1000


def num_val_samples():
    return 100


def sample_dims():
    return (2 * sequence_length, )


def vocab_size():
    return _vocab_size
