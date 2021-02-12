import configparser
import os
import random

import numpy as np

# ----------------------------------------------
# Configuration
# ----------------------------------------------

# Load config file
config_file = os.getenv('LBANN_COMMUNITYGAN_CONFIG_FILE')
if not config_file:
    raise RuntimeError(
        'No configuration file in '
        'LBANN_COMMUNITYGAN_CONFIG_FILE environment variable')
if not os.path.exists(config_file):
    raise FileNotFoundError(f'Could not find config file at {config_file}')
config = configparser.ConfigParser()
config.read(config_file)

# Options from config file
motif_file = config.get('Motifs', 'file')
motif_size = config.getint('Motifs', 'motif_size')
walk_file = config.get('Walks', 'file')
walk_length = config.getint('Walks', 'walk_length')
mini_batch_size = config.getint('Embeddings', 'mini_batch_size')
sgd_steps_per_epoch = config.getint('Embeddings', 'sgd_steps_per_epoch')
assert (motif_file and motif_size>0
        and walk_file and walk_length>0
        and mini_batch_size>0 and sgd_steps_per_epoch>0), \
    f'Invalid config in {config_file}'

# Configure RNG
rng_pid = None
def initialize_rng():
    """Initialize NumPy random seed if needed.

    Seed should be initialized independently on each process. We
    reinitialize if we detect a process fork.

    """
    global rng_pid
    if rng_pid != os.getpid():
        rng_pid = os.getpid()
        random.seed()
        np.random.seed()

# ----------------------------------------------
# Load data
# ----------------------------------------------

motifs = np.loadtxt(motif_file)
walks = np.loadtxt(walk_file)
assert motifs.shape[1] == motif_size+1, "motif size doesn't match motif data"
assert walks.shape[1] == walk_length, "walk length doesn't match walk data"

# ----------------------------------------------
# Sample access functions
# ----------------------------------------------

def num_samples():
    return mini_batch_size * sgd_steps_per_epoch

def sample_dims():
    return (motifs.shape[1]-1 + walks.shape[1],)

def get_sample(_):
    initialize_rng()
    motif = motifs[random.randrange(len(motifs))][1:].copy()
    np.random.shuffle(motif)
    walk = walks[random.randrange(len(walks))]
    return np.concatenate((motif, walk))
