"""Random walk dataset.

This is intended to be imported by the Python data reader and used to
obtain data samples.

"""
import os.path
import numpy as np
import utils.snap
root_dir = os.path.dirname(os.path.realpath(__file__))

# Options
graph_name = 'blog'
graph_file = os.path.join(
    root_dir, 'largescale_node2vec', 'evaluation', 'dataset',
    'blog', 'edges_0based'
)
walk_length = 80
walk_context_length = 10
walks_per_node = 4
return_param = 1.0
inout_param = 1.0
directed = False
weighted = False

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

# Sample access functions
def get_sample(index):
    contexts_per_walk = walk_length - walk_context_length + 1
    walk_index, context_index = divmod(index, contexts_per_walk)
    return walks[walk_index,
                 context_index:context_index+walk_context_length]
def num_samples():
    num_walks = walks.shape[0]
    contexts_per_walk = walk_length - walk_context_length + 1
    return num_walks * contexts_per_walk
def sample_dims():
    return (walk_context_length,)

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
