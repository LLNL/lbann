"""Dataset for offline random walks.

This script loads random walks that have been generated offline. It is
intended to be imported by the Python data reader and used in a
Skip-Gram algorithm. Data samples consists of a random walk and
negative samples.

"""
import os.path
import sys
import numpy as np

# Local imports
root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_dir)
import utils.snap

# Graph options
graph_name = 'facebook'
directed = False
weighted = False

# Random walk options
walk_length = 80        # Length of each random walk
walk_context_size = 10  # Sequence length for Skip-gram
walks_per_node = 10     # Number of random walks starting on each node
return_param = 1.0      # p-parameter
inout_param = 1.0       # q-parameter

# Negative sampling options
num_negative_samples = 5
noise_distribution_exp = 0.75   # Exponent to convert unigram
                                # distribution to noise distribution

# Download graph and perform random walk, if needed
graph_file = utils.snap.download_graph(graph_name)
data_dir = os.path.dirname(graph_file)
walk_file = os.path.join(data_dir, 'walk.txt')
if not os.path.isfile(walk_file):
    utils.snap.node2vec_walk(
        graph_file,
        walk_file,
        walk_length,
        walks_per_node,
        return_param,
        inout_param,
        directed,
        weighted)

# Load random walks from file
walks = np.loadtxt(walk_file, dtype=int)
assert walks.shape[1] == walk_length, \
    ('Random walks in {} have length {}, but expected a walk length of {}'
     .format(walk_file, walks.shape[1], walk_length))

# Generate noise distribution for negative sampling, if needed
unigram_distribution_file = os.path.join(data_dir, 'unigram_distribution.npy')
noise_distribution_cdf_file = os.path.join(data_dir, 'noise_distribution_cdf.npy')
if os.path.isfile(noise_distribution_cdf_file):
    noise_distribution_cdf = np.load(noise_distribution_cdf_file)
else:
    counts = np.bincount(walks.reshape(-1))
    unigram_distribution = counts / walks.size
    noise_counts = counts ** noise_distribution_exp
    noise_distribution_cdf = np.cumsum(noise_counts)
    noise_distribution_cdf /= noise_distribution_cdf[-1]
    np.save(unigram_distribution_file, unigram_distribution)
    np.save(noise_distribution_cdf_file, noise_distribution_cdf)

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

    # Get context window from random walk
    contexts_per_walk = walk_length - walk_context_size + 1
    walk_index, context_index = divmod(index, contexts_per_walk)
    walk_context = walks[walk_index,
                         context_index:context_index+walk_context_size]

    # Generate negative samples
    negative_samples = np.searchsorted(noise_distribution_cdf,
                                       np.random.rand(num_negative_samples))

    # Return concatenated arrays
    return np.concatenate((negative_samples, walk_context))

def num_samples():
    """Number of samples in dataset."""
    num_walks = walks.shape[0]
    contexts_per_walk = walk_length - walk_context_size + 1
    return num_walks * contexts_per_walk

def sample_dims():
    """Dimensions of a data sample."""
    return (walk_context_size + num_negative_samples,)
