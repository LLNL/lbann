"""
The Pile dataset, stored as pre-tokenized binary files for optimized processing.
"""
import os
import os.path

import numpy as np
# ----------------------------------------------
# Options
# ----------------------------------------------

sequence_length = int(os.getenv('THE_PILE_SEQUENCE_LENGTH', default='512'))

# ----------------------------------------------
# Setup
# ----------------------------------------------

# Load the datasets
data_dir = os.getenv('THE_PILE_DATA_DIR',
                     '/p/vast1/data/datasets/the-pile-pretokenized')
dataset_train = np.memmap(os.path.join(data_dir, 'train.bin'),
                          dtype=np.uint16,
                          mode='r')
sample_lengths_train = np.fromfile(os.path.join(data_dir, 'train-seqlen.bin'),
                                   dtype=np.uint32).astype(np.uint64)
sample_offsets_train = np.zeros_like(sample_lengths_train)
sample_offsets_train[1:] = np.cumsum(sample_lengths_train)[:-1]
dataset_val = np.memmap(os.path.join(data_dir, 'val.bin'),
                        dtype=np.uint16,
                        mode='r')
sample_lengths_val = np.fromfile(os.path.join(data_dir, 'val-seqlen.bin'),
                                 dtype=np.uint32).astype(np.uint64)
sample_offsets_val = np.zeros_like(sample_lengths_val)
sample_offsets_val[1:] = np.cumsum(sample_lengths_val)[:-1]

# Uses the definition from the GPT-NeoX-20B tokenizer
pad_index = 1  # '<|padding|>'
_vocab_size = 50277

# ----------------------------------------------
# Sample access functions
# ----------------------------------------------


def trim_and_pad(sample, random: bool):
    # Trim long sequences
    if len(sample) > sequence_length:
        if random:
            pos = np.random.rand()
            offset = (len(sample) - sequence_length + 1) * pos
            offset = int(np.floor(offset))
            sample = sample[offset:offset + sequence_length]
        else:
            sample = sample[0:sequence_length]

    # Left-pad short sequences
    if len(sample) < sequence_length:
        sample_pad = np.full(sequence_length, pad_index, dtype=np.int32)
        if len(sample) > 0:
            sample_pad[-len(sample):] = sample
        return sample_pad

    return sample


def get_train_sample(index: int):
    sample = np.copy(
        dataset_train[sample_offsets_train[index]:sample_offsets_train[index] +
                      sample_lengths_train[index]]).astype(np.int32)
    return trim_and_pad(sample, True)


def get_val_sample(index):
    sample = np.copy(
        dataset_val[sample_offsets_val[index]:sample_offsets_val[index] +
                    sample_lengths_val[index]]).astype(np.int32)
    return trim_and_pad(sample, False)


def num_train_samples():
    return sample_lengths_train.shape[0]


def num_val_samples():
    return sample_lengths_val.shape[0]


def sample_dims():
    return (sequence_length, )


def vocab_size():
    return _vocab_size


if __name__ == '__main__':
    print('Training samples:', num_train_samples())
    print('Validation samples:', num_val_samples())
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(
        os.path.join(data_dir, '20B_tokenizer.json'))
    print('Training sample 101:')
    print(tokenizer.decode(get_train_sample(101)))
    print('Validation sample 233:')
    print(tokenizer.decode(get_val_sample(233)))
