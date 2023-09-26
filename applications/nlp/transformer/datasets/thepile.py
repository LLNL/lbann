"""The Pile dataset."""
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

sequence_length = 512

# ----------------------------------------------
# Setup
# ----------------------------------------------

# Load the dataset with HuggingFace datasets
data_dir = '/p/vast1/data/datasets/the-pile-huggingface'
dataset_train, dataset_val = load_dataset(os.path.join(data_dir, 'pile.py'),
                                          'all',
                                          split=('train', 'validation'),
                                          cache_dir=os.path.join(
                                              data_dir, 'cache'))

# Use the WMT 2016 tokenizer (?)
tokenizer = Tokenizer.from_file(
    os.path.join(utils.paths.wmt_dir(), 'tokenizer-wmt16.json'))
pad_index = tokenizer.token_to_id('<pad>')
bos_index = tokenizer.token_to_id('<s>')
eos_index = tokenizer.token_to_id('</s>')

# ----------------------------------------------
# Tokenization
# ----------------------------------------------


def tokenize(text):
    """Convert string to list of token indices.

    Use byte-pair encoding trained on WMT-16. We add BOS and EOS tokens.
    """
    return tokenizer.encode('<s>' + text + '</s>').ids


def detokenize(indices):
    """Convert token indices to string.

    Stops at the first EOS token. All other special tokens are ignored.
    """
    return tokenizer.decode(indices,
                            skip_special_tokens=True).replace(' ##', '')


# ----------------------------------------------
# Sample access functions
# ----------------------------------------------


def get_train_sample(index):
    """Token indices for a data sample from the training set.

    The English and German text samples are tokenized,
    padded/subsampled to sequence_length tokens, and concatenated.

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
        sample_pad[-len(sample):] = sample
        return sample_pad

    return sample


def get_val_sample(index):
    """Token indices for a data sample from the validation set."""
    text = dataset_train[index]['text']
    return tokenize(text)


def num_train_samples():
    return len(dataset_train)


def num_val_samples():
    return len(dataset_val)


def sample_dims():
    return (sequence_length, )


def vocab_size():
    return tokenizer.get_vocab_size()
