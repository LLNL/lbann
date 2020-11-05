"""Dataset for offline random walks.

This script loads random walks that have been generated offline. It is
intended to be imported by the Python data reader and used in a
Skip-Gram algorithm. Data samples consists of a random walk and
negative samples.

"""
import configparser
import os
import os.path
import numpy as np
import pandas as pd

# Load config file
config_file = os.getenv('LBANN_NODE2VEC_CONFIG_FILE')
if not config_file:
    raise RuntimeError(
        'No configuration file provided in '
        'LBANN_NODE2VEC_CONFIG_FILE environment variable')
if not os.path.exists(config_file):
    raise FileNotFoundError(f'Could not find config file at {config_file}')
config = configparser.ConfigParser()
config.read(config_file)

# Options from config file
num_vertices = config.getint('Graph', 'num_vertices', fallback=0)
walk_file = config.get('Walks', 'file', fallback=None)
walk_length = config.getint('Walks', 'walk_length', fallback=0)
num_walks = config.getint('Walks', 'num_walks', fallback=0)
num_negative_samples = config.getint('Skip-gram', 'num_negative_samples')
noise_distribution_exp = config.getfloat('Skip-gram', 'noise_distribution_exp')

# Load walks from file
### @todo Partial read on each MPI rank
if not walk_file:
    raise RuntimeError(f'No walk file specified in {config_file}')
walks = pd.read_csv(walk_file, delimiter=' ', header=None, dtype=np.int64)
walks = walks.to_numpy()
if not num_walks:
    num_walks = walks.shape[0]
if not walk_length:
    walk_length = walks.shape[1]
if not num_vertices:
    num_vertices = np.amax(walks) + 1
assert walks.shape[0] == num_walks, \
    f'Found {walks.shape[0]} walks in {walk_file}, ' \
    f'but expected {num_walks}'
assert walks.shape[1] == walk_length, \
    f'Found walks of length {walks.shape[1]} in {walk_file}, ' \
    f'but expected a walk length of {walk_length}'
assert num_vertices > 0, \
    f'Random walks in {walk_file} have invalid vertex indices'

# Noise distribution for negative sampling
# Note: We count the number of times each vertex has been visited and
# periodically recompute the noise distribution.
visit_counts = np.ones(num_vertices, dtype=np.int64)
total_visit_count = num_vertices
noise_cdf = np.zeros(num_vertices, dtype=np.float64)
noise_visit_count = 0
def update_noise_distribution():
    global noise_cdf, noise_visit_count
    np.float_power(
        visit_counts,
        noise_distribution_exp,
        out=noise_cdf,
        dtype=np.float64,
    )
    np.cumsum(noise_cdf, out=noise_cdf)
    noise_cdf *= np.float64(1 / noise_cdf[-1])
    noise_visit_count = total_visit_count
update_noise_distribution()

# Need to reseed RNG after forking processes
need_to_seed_rng = True

def get_sample(index):
    """Get a single data sample.

    Consists of a sequence from a random walk and several negative
    samples.

    """

    # Check if RNG needs to be reseeded
    global need_to_seed_rng
    if need_to_seed_rng:
        np.random.seed()
        need_to_seed_rng = False

    # Count number of times each vertex is visited
    # Note: Update noise distribution if there are enough visits.
    global visit_counts, total_visit_count
    visit_counts[walks[index]] += 1
    total_visit_count += len(walks[index])
    if total_visit_count > 2*noise_visit_count:
        update_noise_distribution()

    # Return negative samples and walk
    negative_samples = np.searchsorted(
        noise_cdf,
        np.random.rand(num_negative_samples),
    )
    return np.concatenate((negative_samples, walks[index]))

def num_samples():
    """Number of samples in dataset."""
    return num_walks

def sample_dims():
    """Dimensions of a data sample."""
    return (num_negative_samples + walk_length,)
