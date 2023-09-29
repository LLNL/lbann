"""
Synthetic dataset for benchmarking transformers with source and
target sequences (occurring, e.g., in translation tasks).
"""
import numpy as np

# ----------------------------------------------
# Options
# ----------------------------------------------

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
                             size=(2 * sequence_length + 1, ),
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
    return 100000


def num_val_samples():
    return 100


def sample_dims():
    return (2 * sequence_length + 1, )


def vocab_size():
    return _vocab_size
