"""WMT 2014 dataset for English-German translation."""
import os.path
import sys

import numpy as np
from torch.utils.data import DataLoader
# Local imports
current_file = os.path.realpath(__file__)
root_dir = os.path.dirname(os.path.dirname(current_file))
sys.path.append(root_dir)
import utils.paths

# ----------------------------------------------
# Options
# ----------------------------------------------

# Note: Sequence lengths for WMT 2014 have mean 29.05, standard
# deviation 16.20, and max 484.
sequence_length = 64

# ----------------------------------------------
# Setup
# ----------------------------------------------

pad_index = 0

# ----------------------------------------------
# Tokenization
# ----------------------------------------------

def tokenize(text):
    """Convert string to list of token indices.

    WMT 2014 has already been tokenized with byte-pair encoding. We
    add BOS and EOS tokens.

    """
    indices = [bos_index]
    indices.extend(
        token_indices.get(token, unk_index)
        for token in text.split(' ')
    )
    indices.append(eos_index)
    return indices

def detokenize(indices):
    """Convert token indices to string.

    Stops at the first EOS token. All other special tokens are
    ignored.

    """
    text = ''
    for index in indices:
        if index == eos_index:
            break
        elif index in (unk_index, bos_index, pad_index):
            continue
        else:
            text += f' {tokens[index]}'
    return text

# ----------------------------------------------
# Sample access functions
# ----------------------------------------------

def get_train_sample(index):
    """Token indices for a data sample from the training set.

    The English and German text samples are tokenized,
    padded/subsampled to sequence_length tokens, and concatenated.

    """

    # Tokenize text data


    # Concatenate sequences and return
    sample = np.full(2*sequence_length, pad_index, dtype=int)
    # sample = train_dataset[index]
    # sample[0:len(sample_en)] = sample_en
    # sample[sequence_length:sequence_length+len(sample_de)] = sample_de
    return sample.reshape(-1)

def get_val_sample(index):
    """Token indices for a data sample from the validation set."""
    text = dataset_val[index]
    sample_en = np.full(sequence_length, pad_index, dtype=int)
    sample_de = np.full(sequence_length, pad_index, dtype=int)
    return sample_en, sample_de

def num_train_samples():
    return 10000
def num_val_samples():
    return 1000
def sample_dims():
    return (2*sequence_length+1,)
def vocab_size():
    return 64
