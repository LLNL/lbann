"""WMT 2014 dataset for English-German translation."""
import os.path
import os
import sys

import numpy as np
import torchnlp.datasets

def env2int(env_list, default = -1):
   for e in env_list:
       val = int(os.environ.get(e, -1))
       if val >= 0: return val
   return default

data_size = env2int(['DATA_SIZE'])
# Local imports
current_file = os.path.realpath(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
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

# Load WMT 2014 dataset
data_dir = utils.paths.wmt_dir()
print("Dataset dir", data_dir)
dataset_train, dataset_val = torchnlp.datasets.wmt_dataset(
    directory=data_dir,
    train=True,
    dev=True,
)


if(data_size!=-1):

    dataset_train = dataset_train[:data_size]
    dataset_val = dataset_val[:1024]

# Load token vocabulary
with open(os.path.join(data_dir, 'vocab.bpe.32000')) as f:
    tokens = f.read().splitlines()
tokens.extend(['<unk>', '<s>', '</s>', '<pad>'])
token_indices = dict(zip(tokens, range(len(tokens))))
unk_index = token_indices.get('<unk>', -1)
bos_index = token_indices.get('<s>', -1)
eos_index = token_indices.get('</s>', -1)
pad_index = token_indices.get('<pad>', -1)

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
    text = dataset_train[index]
    sample_en = tokenize(text['en'])
    sample_de = tokenize(text['de'])

    # Randomly subsample sequences if they are too long
    if len(sample_en) > sequence_length or len(sample_de) > sequence_length:
        pos = np.random.rand()
        if len(sample_en) > sequence_length:
            offset = (len(sample_en) - sequence_length + 1) * pos
            offset = int(np.floor(offset))
            sample_en = sample_en[offset:offset+sequence_length]
        if len(sample_de) > sequence_length:
            offset = (len(sample_de) - sequence_length + 1) * pos
            offset = int(np.floor(offset))
            sample_de = sample_de[offset:offset+sequence_length]

    # Concatenate sequences and return
    sample = np.full(2*sequence_length, pad_index, dtype=int)
    sample[0:len(sample_en)] = sample_en
    sample[sequence_length:sequence_length+len(sample_de)] = sample_de
    return sample
def get_test_sample(index):
    """Token indices for a data sample from the training set.

    The English and German text samples are tokenized,
    padded/subsampled to sequence_length tokens, and concatenated.

    """

    # Tokenize text data
    text = dataset_train[index]
    sample_en = tokenize(text['en'])
    sample_de = tokenize(text['de'])

    # Randomly subsample sequences if they are too long
    if len(sample_en) > sequence_length or len(sample_de) > sequence_length:
        pos = np.random.rand()
        if len(sample_en) > sequence_length:
            offset = (len(sample_en) - sequence_length + 1) * pos
            offset = int(np.floor(offset))
            sample_en = sample_en[offset:offset+sequence_length]
        if len(sample_de) > sequence_length:
            offset = (len(sample_de) - sequence_length + 1) * pos
            offset = int(np.floor(offset))
            sample_de = sample_de[offset:offset+sequence_length]

    # Concatenate sequences and return
    sample = np.full(2*sequence_length, pad_index, dtype=int)
    sample[0:len(sample_en)] = sample_en
    sample[sequence_length:sequence_length+len(sample_de)] = sample_de
    return sample
def get_val_sample(index):
    """Token indices for a data sample from the validation set."""
    text = dataset_val[index]
    sample_en = tokenize(text['en'])
    sample_de = tokenize(text['de'])
    return sample_en, sample_de

def num_train_samples():
    return len(dataset_train)
def num_val_samples():
    return len(dataset_val)
def sample_dims():
    return (2*sequence_length+1,)
def vocab_size():
    return len(tokens)
