"""The Pile dataset."""
import os
import os.path
import sys

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer

# Local imports
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_dir, '..', '..'))
import utils.paths

# ----------------------------------------------
# Options
# ----------------------------------------------

sequence_length = int(os.getenv('THE_PILE_SEQUENCE_LENGTH', default='512'))

# ----------------------------------------------
# Setup
# ----------------------------------------------

# Load the dataset with HuggingFace datasets
data_dir = os.getenv('THE_PILE_DATA_DIR',
                     '/p/vast1/data/datasets/the-pile-huggingface')
dataset_train, dataset_val = load_dataset(os.path.join(data_dir, 'pile.py'),
                                          'all',
                                          split=('train', 'validation'),
                                          cache_dir=os.path.join(
                                              data_dir, 'cache'))

# Use the GPT-NeoX-20B tokenizer
tokenizer = Tokenizer.from_file(os.path.join(data_dir, '20B_tokenizer.json'))
pad_index = tokenizer.token_to_id('<|padding|>')

# ----------------------------------------------
# Tokenization
# ----------------------------------------------


def tokenize(text):
    """Convert string to list of token indices.

    Use byte-pair encoding trained on The Pile.
    """
    return tokenizer.encode(text).ids


def detokenize(indices):
    """Convert token indices to string.

    Stops at the first EOS token. All other special tokens are ignored.
    """
    return tokenizer.decode(indices)


# ----------------------------------------------
# Sample access functions
# ----------------------------------------------


def get_train_sample(index):
    """Token indices for a data sample from the training set.

    The text samples are tokenized and padded/subsampled to sequence_length
    tokens.
    """

    # Tokenize text data
    text = dataset_train[index]['text']
    sample = tokenize(text)

    # Randomly subsample sequences if they are too long
    if len(sample) > sequence_length:
        pos = np.random.rand()
        offset = (len(sample) - sequence_length + 1) * pos
        offset = int(np.floor(offset))
        sample = sample[offset:offset + sequence_length]

    # Left-pad sequences if they are too short
    if len(sample) < sequence_length:
        sample_pad = np.full(sequence_length, pad_index, dtype=int)
        if len(sample) > 0:
            sample_pad[-len(sample):] = sample
        return sample_pad

    return sample


def get_val_sample(index):
    """Token indices for a data sample from the validation set."""
    text = dataset_train[index]['text']
    tokenized = tokenize(text)

    # Trim long sequences, left-pad short sequences
    if len(tokenized) > sequence_length:
        tokenized = tokenized[0:sequence_length]
    if len(tokenized) < sequence_length:
        sample_pad = np.full(sequence_length, pad_index, dtype=np.int32)
        if len(tokenized) > 0:
            sample_pad[-len(tokenized):] = tokenized
        return sample_pad

    return tokenized


def num_train_samples():
    return len(dataset_train)


def num_val_samples():
    return len(dataset_val)


def sample_dims():
    return (sequence_length, )


def vocab_size():
    return tokenizer.get_vocab_size()
