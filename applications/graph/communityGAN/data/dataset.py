import os
import random

# ----------------------------------------------
# Configuration
# ----------------------------------------------

# Hard-coded options
### @todo Specify with config file
motif_size = 4
walk_length = 20
epoch_size = 51200

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
# Sample access functions
# ----------------------------------------------

def num_samples():
    return epoch_size

def sample_dims():
    return (motif_size+walk_length,)

def get_sample(_):
    ### @todo Real data
    initialize_rng()
    motif = random.sample(range(40,80), motif_size)
    walk = random.sample(range(0,80), walk_length)
    return motif + walk
