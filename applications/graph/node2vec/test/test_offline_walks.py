import os
import os.path
import random
import sys

# Local paths
root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.join(root_dir, 'data'))

def test_dataset():
    """Make sure offline walk ingestion produces sane data.

    The LBANN_NODE2VEC_CONFIG_FILE environment variable must be set.

    """
    import offline_walks

    # Check graph size
    num_vertices = offline_walks.num_vertices
    assert num_vertices >= 0, 'Negative graph size'
    assert num_vertices != 0, 'Graph has no vertices'

    # Check number of samples
    num_samples = offline_walks.num_samples()
    assert num_samples >= 0, 'Invalid number of data samples'

    # Check sample dimensions
    sample_dims = offline_walks.sample_dims()
    assert len(sample_dims) == 1, 'Unexpected dimensions for data sample'
    assert sample_dims[0] > 0, 'Invalid dimensions for data sample'

    # Check samples
    for _ in range(1000):
        sample = offline_walks.get_sample(0)
        assert sample.shape == sample_dims, 'Unexpected dimensions for data sample'
        for node in sample:
            assert 0 <= node < num_vertices, \
                'Invalid graph node ID in data sample'

def profile_dataset():
    """Profile offline walk ingestion.

    The LBANN_NODE2VEC_CONFIG_FILE environment variable must be set. A
    cProfile file is output to benchmark_dataset.prof in the
    application dir, which can be visualized with SnakeViz.

    """
    import cProfile
    import offline_walks
    output_file = os.path.join(root_dir, 'benchmark_dataset.prof')
    num_iters = 1000
    def func():
        for _ in range(1000):
            offline_walks.get_sample(0)
    cProfile.runctx('func()', globals(), locals(), filename=output_file)

if __name__ == '__main__':
    test_dataset()
