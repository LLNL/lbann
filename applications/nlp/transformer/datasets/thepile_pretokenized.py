"""
The Pile dataset, stored as pre-tokenized binary files for optimized processing.
"""
import os
import os.path
import pickle

import numpy as np
# ----------------------------------------------
# Options
# ----------------------------------------------

sequence_length = int(os.getenv('THE_PILE_SEQUENCE_LENGTH', default='512'))

# ----------------------------------------------
# Setup
# ----------------------------------------------

# Load the datasets
data_dir = os.getenv(
    'THE_PILE_DATA_DIR',
    '/p/vast1/data/datasets/the-pile-huggingface/pretokenized')
dataset_train = np.memmap(os.path.join(data_dir, 'train.bin'),
                          dtype=np.uint16,
                          mode='r').reshape(-1, sequence_length)
with open(os.path.join(data_dir, 'val.bin'), 'rb') as fp:
    dataset_val = pickle.load(fp)

# Uses the definition from the GPT-NeoX-20B tokenizer
pad_index = 1  # '<|padding|>'
_vocab_size = 50277

# ----------------------------------------------
# Sample access functions
# ----------------------------------------------


def get_train_sample(index):
    return dataset_train[index]


def get_val_sample(index):
    return dataset_val[index]


def num_train_samples():
    return dataset_train.shape[0]


def num_val_samples():
    return len(dataset_val)


def sample_dims():
    return (sequence_length, )


def vocab_size():
    return _vocab_size


if __name__ == '__main__':
    print('Training samples:', num_train_samples())
    print('Validation samples:', num_val_samples())
    from tokenizers import Tokenizer
    tok_dir = os.path.join(data_dir, '..')
    tokenizer = Tokenizer.from_file(os.path.join(tok_dir,
                                                 '20B_tokenizer.json'))
    print('Training sample 101:')
    print(tokenizer.decode(get_train_sample(101)))
    print('Validation sample 233:')
    print(tokenizer.decode(get_val_sample(233)))
