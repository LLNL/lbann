"""WMT 2016 dataset for English-German translation."""
import os.path
import sys

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer

# Local imports
current_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.join(current_dir, '..', '..')
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

# Load WMT 2016 dataset and tokenizer.
data_dir = utils.paths.wmt_dir()
dataset_train, dataset_val = load_dataset(os.path.join(data_dir, 'wmt16.py'),
                                          'de-en',
                                          split=('train', 'validation'),
                                          cache_dir=os.path.join(
                                              data_dir, 'cache'))
tokenizer = Tokenizer.from_file(os.path.join(data_dir, 'tokenizer-wmt16.json'))
pad_index = tokenizer.token_to_id('<pad>')
bos_index = tokenizer.token_to_id('<s>')
eos_index = tokenizer.token_to_id('</s>')

# ----------------------------------------------
# Tokenization
# ----------------------------------------------


def tokenize(text):
    """Convert string to list of token indices.

    WMT 2016 has already been tokenized with byte-pair encoding. We
    add BOS and EOS tokens.

    """
    return tokenizer.encode('<s>' + text + '</s>').ids


def detokenize(indices):
    """Convert token indices to string.

    Stops at the first EOS token. All other special tokens are
    ignored.

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
    text = dataset_train[index]['translation']
    sample_en = tokenize(text['en'])
    sample_de = tokenize(text['de'])

    # Randomly subsample sequences if they are too long
    if len(sample_en) > sequence_length or len(sample_de) > sequence_length:
        pos = np.random.rand()
        if len(sample_en) > sequence_length:
            offset = (len(sample_en) - sequence_length + 1) * pos
            offset = int(np.floor(offset))
            sample_en = sample_en[offset:offset + sequence_length]
        if len(sample_de) > sequence_length:
            offset = (len(sample_de) - sequence_length + 1) * pos
            offset = int(np.floor(offset))
            sample_de = sample_de[offset:offset + sequence_length]

    # Concatenate sequences and return
    sample = np.full(2 * sequence_length, pad_index, dtype=int)
    sample[0:len(sample_en)] = sample_en
    sample[sequence_length:sequence_length + len(sample_de)] = sample_de
    return sample


def get_val_sample(index):
    """Token indices for a data sample from the validation set."""
    text = dataset_val[index]['translation']
    sample_en = tokenize(text['en'])
    sample_de = tokenize(text['de'])
    return sample_en, sample_de


def num_train_samples():
    return len(dataset_train)


def num_val_samples():
    return len(dataset_val)


def sample_dims():
    return (2 * sequence_length + 1, )


def vocab_size():
    return tokenizer.get_vocab_size()
