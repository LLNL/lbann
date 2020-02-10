import os.path
import sys

import numpy as np
import torchnlp.datasets

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
pad_index = 0

# ----------------------------------------------
# Setup
# ----------------------------------------------

# Load WMT 2014 dataset
data_dir = utils.paths.wmt_dir()
dataset_train, dataset_val = torchnlp.datasets.wmt_dataset(
    directory=data_dir,
    train=True,
    dev=True,
)

# Load token vocabulary
tokens = ['<pad>']
with open(os.path.join(data_dir, 'vocab.bpe.32000')) as f:
    tokens.extend(f.read().splitlines())
token_indices = dict(zip(tokens, range(len(tokens))))

# ----------------------------------------------
# Sample access functions
# ----------------------------------------------

def get_train_sample(index):
    """Token indices for a data sample from the training set.

    The English and German text samples are each padded/subsampled to
    sequence_length tokens and concatenated with a pad token
    separating them. This separating pad token is there to simplify
    the input to the transformer decoder, which is shifted right.

    """

    # Get text data
    text = dataset_train[index]
    text_en = text['en'].split(' ')
    text_de = text['de'].split(' ')

    # Randomly subsample text data if it's too long
    if len(text_en) > sequence_length or len(text_de) > sequence_length:
        pos = np.random.rand()
        if len(text_en) > sequence_length:
            offset = (len(text_en) - sequence_length + 1) * pos
            offset = int(np.floor(offset))
            text_en = text_en[offset:offset+sequence_length]
        if len(text_de) > sequence_length:
            offset = (len(text_de) - sequence_length + 1) * pos
            offset = int(np.floor(offset))
            text_de = text_de[offset:offset+sequence_length]

    # Output token indices
    sample = np.full(2*sequence_length+1, pad_index, dtype=int)
    for i, token in enumerate(text_en):
        sample[i] = token_indices.get(token, pad_index)
    for i, token in enumerate(text_de):
        sample[i+sequence_length+1] = token_indices.get(token, pad_index)
    return sample

def get_val_sample(index):
    """Token indices for a data sample from the validation set.

    Only returns English text data.

    """

    # Get text data
    text_en = dataset_val[index].split(' ')

    # Truncate text data if it's too long
    if len(text_en) > sequence_length:
        text_en = text_en[:sequence_length]

    # Output token indices
    sample = np.full(sequence_length, pad_index, dtype=int)
    for i, token in enumerate(text_en):
        sample[i] = token_indices.get(token, pad_index)
    return sample

def num_train_samples():
    return len(dataset_train)
def num_val_samples():
    return len(dataset_val)
def sample_dims():
    return (2*sequence_length+1,)
def vocab_size():
    return len(tokens)
