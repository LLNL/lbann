"""
The QM9 dataset, stored as pre-tokenized binary files for optimized processing.
"""

import os
import os.path

import numpy as np
from pretokenize.SMILES_tokenizer import MolTokenizer

sequence_length = int(os.getenv("QM9_SEQUENCE_LENGTH", default="32"))

# ----------------------------------------------
# Setup
# ----------------------------------------------

# Load the datasets
data_dir = os.getenv("QM9_DATA_DIR", "/p/vast1/lbann/datasets/FLASK/QM9")

tokenizer = MolTokenizer("SMILES_vocab.json")
tokenizer.load_vocab_file()

dataset_train = np.load(os.path.join(data_dir, "QM9_Pretokenized.npy"))

_vocab_size = 46


# ----------------------------------------------
# Sample access functions
# ----------------------------------------------


def num_train_samples():
    return dataset_train.shape[0]


def get_train_sample(i):
    data = dataset_train[i]
    return data


def sample_dims():
    return (2 * sequence_length + 1,)


def vocab_size():
    return _vocab_size


if __name__ == "__main__":
    print("Training samples:", num_train_samples())
    print("Training sample 101:")
    print(get_train_sample(101))
