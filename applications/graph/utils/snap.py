import os
import os.path
import urllib.request
import gzip
import subprocess

import snap

# Root directory for LBANN graph application
_root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def download_graph(name='ego-Facebook',
                   data_dir=None):
    """Download graph edgelist file from Stanford SNAP website.

    Args:
        name (str): Name of graph.
        data_dir (str, optional): Directory for downloading data.

    Returns:
        str: Uncompressed edge list file.

    """

    # Graphs from Stanford SNAP
    download_urls = {
        'ego-Facebook': 'http://snap.stanford.edu/data/facebook_combined.txt.gz',
    }

    # Make data directory if needed
    if not data_dir:
        data_dir = os.path.join(_root_dir, 'data', name)
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    data_dir = os.path.realpath(data_dir)

    # Download and uncompress graph file
    gzip_file = os.path.join(data_dir, 'graph.txt.gz')
    txt_file = os.path.join(data_dir, 'graph.txt')
    urllib.request.urlretrieve(download_urls[name],
                               filename=gzip_file)
    with gzip.open(gzip_file, 'rb') as in_file:
        with open(txt_file, 'wb') as out_file:
            out_file.write(in_file.read())

    return txt_file


def node2vec_walk(graph_file,
                  walk_file,
                  walk_length,
                  walks_per_node,
                  return_param=1.0,
                  inout_param=1.0,
                  directed=False,
                  weighted=False,
                  verbose=False):
    """Perform random walk on graph for node2vec.

    See https://cs.stanford.edu/~jure/pubs/node2vec-kdd16.pdf

    Args:
        graph_file (str): Uncompressed edge list file.
        walk_file (str): File where random walks will be saved.
        walk_length (int): Walk length.
        walks_per_node (int): Number of walks per graph vertex.
        return_param (float, optional): p-parameter for random walk
            (default: 1.0).
        inout_param (float, optional): q-parameter for random walk
            (default: 1.0).
        directed (bool, optional): Graph is directed (default: False).
        weighted (bool, optional): Graph is weighted (default: False).
        verbose (bool, optional): Verbose output (default: False).

    """

    # Check executable
    node2vec_exe = os.path.join(_root_dir, 'snap', 'examples',
                                'node2vec', 'node2vec')
    if not os.path.isfile(node2vec_exe):
        raise FileNotFoundError(
            'Could not find node2vec executable at {}. '
            'Has SNAP been built?'
            .format(node2vec_exe)
        )

    # Construct invocation
    command = [
        node2vec_exe,
        '-i:{}'.format(graph_file),
        '-o:{}'.format(walk_file),
        '-d:-1',
        '-l:{}'.format(walk_length),
        '-r:{}'.format(walks_per_node),
        '-k:-1',
        '-e:-1',
        '-p:{}'.format(return_param),
        '-q:{}'.format(inout_param),
        '-ow',
    ]
    if verbose:
        command.append('-v')
    if directed:
        command.append('-dr')
    if weighted:
        command.append('-w')

    # Run executable
    return subprocess.call(command)
