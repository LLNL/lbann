"""
Synthetic dataset for benchmarking transformers on autoregressive tasks (e.g.,
in causal language models).
"""
import numpy as np

# ----------------------------------------------
# Options
# ----------------------------------------------

sequence_length = 512
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
                             size=(sequence_length, ),
                             dtype=np.int32)


def get_val_sample(index):
    return np.random.randint(0,
                             _vocab_size,
                             size=(sequence_length, ),
                             dtype=np.int32)


def num_train_samples():
    return 100000


def num_val_samples():
    return 100


def sample_dims():
    return (sequence_length, )


def vocab_size():
    return _vocab_size
