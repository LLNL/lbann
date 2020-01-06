"""Random walk dataset.

This is intended to be imported by the Python data reader and used to
obtain data samples.

"""
import os.path
import numpy as np
import utils.snap
root_dir = os.path.dirname(os.path.realpath(__file__))

# Graph options
graph_name = 'blog'
graph_file = os.path.join(
    root_dir, 'largescale_node2vec', 'evaluation', 'dataset',
    'blog', 'edges_0based'
)
# graph_file = os.path.join(root_dir, 'data', graph_name, 'graph.txt')
directed = False
weighted = False

# Random walk options
walk_length = 80        # Length of each random walk
walk_context_length = 10    # Sequence length for Skip-gram
walks_per_node = 10     # Number of random walks starting on each node
return_param = 0.25     # p-parameter
inout_param = 0.25      # q-parameter

# Negative sampling options
num_negative_samples = 5
noise_distribution_exp = 0.75   # Exponent to convert unigram
                                # distribution to noise distribution

# Download graph and perform random walk, if needed
data_dir = os.path.join(root_dir, 'data', graph_name)
walk_file = os.path.join(data_dir, 'walk.txt')
if not os.path.isfile(graph_file):
    utils.snap.download_graph(graph_name, graph_file)
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
    contexts_per_walk = walk_length - walk_context_length + 1
    walk_index, context_index = divmod(index, contexts_per_walk)
    walk_context = walks[walk_index,
                         context_index:context_index+walk_context_length]

    # Generate negative samples
    negative_samples = np.searchsorted(noise_distribution_cdf,
                                       np.random.rand(num_negative_samples))

    # Return concatenated arrays
    return np.concatenate((negative_samples, walk_context))

def num_samples():
    """Number of samples in dataset."""
    num_walks = walks.shape[0]
    contexts_per_walk = walk_length - walk_context_length + 1
    return num_walks * contexts_per_walk

def sample_dims():
    """Dimensions of a data sample."""
    return (walk_context_length + num_negative_samples,)

def max_graph_node_id(graph_file=graph_file):
    """Largest node ID in graph.

    Nodes should be numbered consecutively from 0 to
    (num_graph_nodes-1). If there are any gaps in the IDs, then
    unnecessary memory will be allocated. If any IDs are negative,
    there may be mysterious errors.

    Args:
        graph_file (str): Uncompressed edge list file.

    Returns:
        int: Largest node ID in graph.

    """
    max_id = -1
    with open(graph_file) as f:
        for line in f:
            line = line.split('#')[0]
            line = line.split()
            if len(line) >= 2:
                max_id = max(max_id, int(line[0]))
                max_id = max(max_id, int(line[1]))
    if max_id < 0:
        raise RuntimeError('Graph has no non-negative node IDs')
    return max_id
