"""Dataset for offline random walks.

This script loads graph walks that have been generated offline. It is
intended to be imported by the Python data reader and used in a
Skip-Gram algorithm. Data samples are sequences of vertex indices,
corresponding to a graph walk and negative samples.

"""
import configparser
import threading
import os
import os.path
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
# Sample generator
# ----------------------------------------------

class SampleIterator:
    """Iterator class to produces data samples.

    A data sample consists of a graph walk and several negative
    samples. Data samples are produced in batches to amortize Python
    and I/O overheads.

    """

    def __init__(
            self,
            walk_file,
            walk_length,
            num_vertices,
            num_negative_samples,
            noise_distribution_exp,
            batch_size,
        ):

        # Options
        self.walk_file = walk_file
        self.walk_length = walk_length
        self.num_vertices = num_vertices
        self.num_negative_samples = num_negative_samples
        self.noise_distribution_exp = noise_distribution_exp
        self.batch_size = batch_size

        # Cache for walk data
        self.walk_batches = iter(())
        self.next_walk_batch = None
        self.next_walk_batch_thread = None

        # Cache for batched sample generation
        self.batch = np.zeros((self.batch_size, sample_dims()[0]), dtype=np.float32)
        self.batch_rows = []

        # Negative sampling distribution
        self.noise_cdf = np.zeros(num_vertices, dtype=np.float64)
        self.noise_visit_count = 0
        self.visit_counts = np.ones(self.num_vertices, dtype=np.int64)
        self.total_visit_count = self.num_vertices
        self._update_noise_distribution()

    def __iter__(self):
        return self

    def __next__(self):
        if not self.batch_rows:
            self._generate_batch(
                out=self.batch,
                rows=self.batch_rows,
            )
        return self.batch[self.batch_rows.pop()]

    def _read_walks(self):
        """Read walk data from walks file.

        Results are stored in `self.next_walk_batch`. Each MPI process
        reads a different subset of the file.

        """

        # Read walks from file
        try:
            walks = next(self.walk_batches)
        except StopIteration:
            self.walk_batches = pd.read_csv(
                self.walk_file,
                delimiter=' ',
                header=None,
                dtype=np.int64,
                skiprows=(lambda row : (row + mpi_rank) % mpi_size),
                keep_default_na=False,
                iterator=True,
                chunksize=self.batch_size,
                compression=None,
                memory_map=True,
            )
            walks = next(self.walk_batches)
        walks = walks.to_numpy(dtype=np.int64)

        # Check that walks are valid
        assert walks.shape[0] > 0, \
            f'Did not load any walks from {walk_file}'
        assert walks.shape[1] == self.walk_length, \
            f'Found walks of length {walks.shape[1]} in {walk_file}, ' \
            f'but expected a walk length of {self.walk_length}'

        # Store walks in member variable
        self.next_walk_batch = walks

    def _generate_batch(self, out, rows):
        """Produce a batch of data samples.

        Batch data is output to `out` and valid matrix rows are output
        to `rows`. `out` should be a `np.ndarray` with type
        `np.float32` and dimensions `batch_size` x `sample_size`.
        `rows` should be a `list`.

        Walk data is read from the walk file in a background thread.

        """
        initialize_rng()

        # Get walks and start reading next batch in background
        if self.next_walk_batch_thread is None:
            self._read_walks()
        else:
            self.next_walk_batch_thread.join()
        walks = self.next_walk_batch
        batch_size = min(walks.shape[0], out.shape[0])
        self.next_walk_batch_thread = threading.Thread(target=self._read_walks)
        self.next_walk_batch_thread.start()

        # Update negative sampling distribution, if needed
        self._record_walk_visits(walks)
        if self.total_visit_count > 2*self.noise_visit_count:
            self._update_noise_distribution()

        # Generate negative samples
        rands = np.random.uniform(size=(batch_size, self.num_negative_samples))
        negative_samples = np.searchsorted(self.noise_cdf, rands)

        # Populate output matrix with data samples
        col0 = 0
        col1 = self.num_negative_samples
        col2 = self.num_negative_samples + self.walk_length
        out[:batch_size,col0:col1] = negative_samples
        out[:batch_size,col1:col2] = walks[:batch_size]

        # Construct list of valid rows in output matrix
        rows.clear()
        rows.extend(range(batch_size))
        np.random.shuffle(rows)

    def _record_walk_visits(self, walks):
        """Record visits to vertices in a graph walk."""
        # Note: Doesn't correctly handle case where vertex is visited
        # multiple times
        self.visit_counts[walks] += np.int64(1)
        self.total_visit_count += walks.size

    def _update_noise_distribution(self):
        """Recomputes negative sampling probability distribution."""
        np.float_power(
            self.visit_counts,
            self.noise_distribution_exp,
            out=self.noise_cdf,
            dtype=np.float64,
        )
        np.cumsum(self.noise_cdf, out=self.noise_cdf)
        self.noise_cdf *= np.float64(1 / self.noise_cdf[-1])
        self.noise_visit_count = self.total_visit_count

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
    if samples is None:
        samples = SampleIterator(
            walk_file=walk_file,
            walk_length=walk_length,
            num_vertices=num_vertices,
            num_negative_samples=num_negative_samples,
            noise_distribution_exp=noise_distribution_exp,
            batch_size=4096,
        )
    return next(samples)
