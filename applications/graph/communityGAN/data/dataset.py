import configparser
import os
import random

import numpy as np

# ----------------------------------------------
# Configuration
# ----------------------------------------------

# Hard-coded options
### @todo Get options from config file
motif_size = 4
epoch_size = 51200

# Load config file
config_file = os.getenv('LBANN_COMMUNITYGAN_CONFIG_FILE')
if not config_file:
    raise RuntimeError(
        'No configuration file provided in '
        'LBANN_COMMUNITYGAN_CONFIG_FILE environment variable')
if not os.path.exists(config_file):
    raise FileNotFoundError(f'Could not find config file at {config_file}')
config = configparser.ConfigParser()
config.read(config_file)

# Options from config file
walk_length = config.getint('RW', 'rw_walk_len', fallback=None)
motif_file = config.get('Motifs', 'motif_file', fallback=None)
walk_file = config.get('RW', 'rw_out_filename', fallback=None)

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

# ----------------------------------------------
# Load data
# ----------------------------------------------

motifs = np.loadtxt(motif_file, delimiter=',')
walks = np.loadtxt(walk_file)

# ----------------------------------------------
# Sample access functions
# ----------------------------------------------

def num_samples():
    return epoch_size

def sample_dims():
    return (motifs.shape[1] + walks.shape[1],)

def get_sample(_):
    initialize_rng()
    motif = motifs[random.randrange(len(motifs))]
    walk = walks[random.randrange(len(walks))]
    return np.concatenate((motif, walk))
