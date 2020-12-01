"""Dataset for offline random walks.

This script loads graph walks that have been generated offline. It is
intended to be imported by the Python data reader and used in a
Skip-Gram algorithm. Data samples are sequences of vertex indices,
corresponding to a graph walk and negative samples.

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

# ----------------------------------------------
# Configuration
# ----------------------------------------------

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
epoch_size = config.getint('Skip-gram', 'epoch_size')
num_negative_samples = config.getint('Skip-gram', 'num_negative_samples')
noise_distribution_exp = config.getfloat('Skip-gram', 'noise_distribution_exp')

# Check options
if not walk_file:
    raise RuntimeError(f'No walk file specified in {config_file}')
assert num_negative_samples > 0, \
    f'Invalid number of negative samples ({num_negative_samples})'
assert num_vertices > 0, f'Invalid number of vertices ({num_vertices})'

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
        np.random.seed()

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

# ----------------------------------------------
# Negative sampling
# ----------------------------------------------

# Keep track how often we visit each vertex
# Note: Start with one visit per vertex for Laplace smoothing.
visit_counts = np.ones(num_vertices, dtype=np.int64)
total_visit_count = num_vertices
def record_walk_visits(walks):
    """Record visits to vertices in a graph walk."""
    global visit_counts, total_visit_count
    visit_counts[walks] += np.int64(1)
    total_visit_count += len(walks) * walk_length

# Probability distribution for negative sampling
noise_cdf = np.zeros(num_vertices, dtype=np.float64)
noise_visit_count = 0
def update_noise_distribution():
    """Recomputes negative sampling probability distribution, if needed.

    The distribution is recomputed if enough new vertices have been
    visited.

    """
    global noise_visit_count
    if total_visit_count > 2*noise_visit_count:

        # Update noise distribution if there are enough new visits
        global noise_cdf
        np.float_power(
            visit_counts,
            noise_distribution_exp,
            out=noise_cdf,
            dtype=np.float64,
        )
        np.cumsum(noise_cdf, out=noise_cdf)
        noise_cdf *= np.float64(1 / noise_cdf[-1])
        noise_visit_count = total_visit_count

# Initial negative sampling distribution is uniform
update_noise_distribution()

# ----------------------------------------------
# Sample access functions
# ----------------------------------------------

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

samples = None
def get_sample(*args):
    """Get a single data sample.

    A data sample consists of a graph walk and several negative
    samples. Input arguments are ignored.

    """
    global samples
    if not samples:
        samples = SampleIterator(
            walk_file=walk_file,
            walk_length=walk_length,
            num_negative_samples=num_negative_samples,
            batch_size=4096,
        )
    return next(samples)

class SampleIterator:

    def __init__(
        self,
        walk_file,
        walk_length,
        num_negative_samples,
        batch_size,
    ):

        # Options
        self.walk_file = walk_file
        self.walk_length = walk_length
        self.num_negative_samples = num_negative_samples
        self.batch_size = batch_size

        # Cache for batched sample generation
        self.batch = np.zeros(
            (self.batch_size, sample_dims()[0]),
            dtype=np.float32,
        )
        self.walk_batches = iter(())
        self.batch_pos_list = []

    def __iter__(self):
        return self

    def __next__(self):
        if not self.batch_pos_list:
            self._generate_batch()
        return self.batch[self.batch_pos_list.pop()]

    def __bool__(self):
        return True

    def _generate_batch(self):

        # Read walks from file
        try:
            walk_batch = next(self.walk_batches)
        except StopIteration:
            self.walk_batches = pd.read_csv(
                self.walk_file,
                delimiter=' ',
                header=None,
                dtype=np.int64,
                skiprows=(lambda row : (row + mpi_rank) % mpi_size),
                iterator=True,
                chunksize=self.batch_size,
            )
            walk_batch = next(self.walk_batches)
        walk_batch = walk_batch.to_numpy()
        batch_size = walk_batch.shape[0]
        assert walk_batch.shape[0], \
            f'Did not load any walks from {walk_file}'
        assert walk_batch.shape[1] == self.walk_length, \
            f'Found walks of length {walk_batch.shape[1]} in {walk_file}, ' \
            f'but expected a walk length of {self.walk_length}'

        # Update negative sampling distribution
        record_walk_visits(walk_batch)
        update_noise_distribution()

        # Generate negative samples
        initialize_rng()
        rands = np.random.uniform(size=(batch_size, self.num_negative_samples))
        negative_samples = np.searchsorted(noise_cdf, rands)

        # Cache samples
        self.batch[:batch_size,:self.num_negative_samples] = negative_samples
        self.batch[:batch_size,-self.walk_length:] = walk_batch
        self.batch_pos_list = list(range(batch_size))
        np.random.shuffle(self.batch_pos_list)
