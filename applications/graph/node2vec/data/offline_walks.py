"""Dataset for offline random walks.

This script loads random walks that have been generated offline. It is
intended to be imported by the Python data reader and used in a
Skip-Gram algorithm. Data samples consists of a random walk and
negative samples.

"""
import configparser
import os
import os.path
import subprocess
import sys
import warnings
import numpy as np
import pandas as pd

current_file = os.path.realpath(__file__)
app_dir = os.path.dirname(os.path.dirname(current_file))
sys.path.append(os.path.join(app_dir))
import utils

# Load config file
config_file = os.getenv('LBANN_NODE2VEC_CONFIG_FILE')
if not config_file:
    raise RuntimeError(
        'No configuration file provided in '
        'LBANN_NODE2VEC_CONFIG_FILE environment variable')
if not os.path.exists(config_file):
    raise FileNotFoundError(f'Could not find config file at {config_file}')
config = configparser.ConfigParser()
config.read(os.path.join(app_dir, 'default.config'))
config.read(config_file)

# Options from config file
num_vertices = config.getint('Graph', 'num_vertices', fallback=0)
walk_file = config.get('Walks', 'file', fallback=None)
walk_length = config.getint('Walks', 'walk_length', fallback=0)
num_walks = config.getint('Walks', 'num_walks', fallback=0)
epoch_size = config.getint('Skip-gram', 'epoch_size')
num_negative_samples = config.getint('Skip-gram', 'num_negative_samples')
noise_distribution_exp = config.getfloat('Skip-gram', 'noise_distribution_exp')

# Determine MPI rank
if 'OMPI_COMM_WORLD_RANK' not in os.environ:
    warnings.warn(
        'Could not detect MPI environment variables. '
        'We expect that LBANN is run with '
        'Open MPI or one of its derivatives.',
        warning.RuntimeWarning,
    )
mpi_rank = int(os.getenv('OMPI_COMM_WORLD_RANK', default=0))
mpi_size = int(os.getenv('OMPI_COMM_WORLD_SIZE', default=1))

# Load walks from file
if not walk_file:
    raise RuntimeError(f'No walk file specified in {config_file}')
if not num_walks:
    with open(walk_file, 'r') as f:
        result = subprocess.run(['wc','-l'], stdin=f, stdout=subprocess.PIPE)
    num_walks = int(result.stdout.decode('utf-8'))
local_num_walks = utils.ceildiv(num_walks, mpi_size)
start_walk = local_num_walks * mpi_rank
end_walk = min(local_num_walks * (mpi_rank + 1), num_walks)
local_num_walks = end_walk - start_walk
walks = pd.read_csv(
    walk_file,
    delimiter=' ',
    header=None,
    dtype=np.int64,
    skiprows=start_walk,
    nrows=local_num_walks,
)
walks = walks.to_numpy()
if not walk_length:
    walk_length = walks.shape[1]
if not num_vertices:
    num_vertices = np.amax(walks) + 1
assert walks.shape[0] > 0, f'Did not load any walks from {walk_file}'
assert walks.shape[0] == local_num_walks, \
    f'Read {walks.shape[0]} walks from {walk_file}, ' \
    f'but expected to read {local_num_walks}'
assert walks.shape[1] == walk_length, \
    f'Found walks of length {walks.shape[1]} in {walk_file}, ' \
    f'but expected a walk length of {walk_length}'
assert num_vertices > 0, \
    f'Walks in {walk_file} have invalid vertex indices'

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

def get_sample(_):
    """Get a single data sample.

    Consists of a sequence from a random walk and several negative
    samples.

    """

    # Check if RNG needs to be reseeded
    global need_to_seed_rng
    if need_to_seed_rng:
        np.random.seed()
        need_to_seed_rng = False

    # Randomly choose walk
    walk = walks[np.random.randint(local_num_walks)]

    # Count number of times each vertex is visited
    # Note: Update noise distribution if there are enough visits.
    global visit_counts, total_visit_count
    visit_counts[walk] += 1
    total_visit_count += len(walk)
    if total_visit_count > 2*noise_visit_count:
        update_noise_distribution()

    # Return negative samples and walk
    negative_samples = np.searchsorted(
        noise_cdf,
        np.random.rand(num_negative_samples),
    )
    return np.concatenate((negative_samples, walk))

def num_samples():
    """Number of samples per "epoch".

    This data reader samples randomly with replacement, so epochs are
    not meaningful. However, it is convenient to group data samples
    into "epochs" so that we don't need to change LBANN's data model.

    """
    return epoch_size

def sample_dims():
    """Dimensions of a data sample."""
    return (num_negative_samples + walk_length,)
