import os.path
import random
import sys

# Local paths
root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_dir)

def test_dataset():
    import dataset

    # Check max node ID
    max_graph_node_id = dataset.max_graph_node_id()
    assert max_graph_node_id >= 0, 'Negative graph node ID'
    assert max_graph_node_id != 0, \
        'Max graph node ID is zero, ' \
        'which implies graph has only one node or node IDs are negative'

    # Check sample dimensions
    sample_dims = dataset.sample_dims()
    assert len(sample_dims) == 1, 'Unexpected dimensions for data sample'
    assert sample_dims[0] > 0, 'Invalid dimensions for data sample'

    # Check number of samples
    num_samples = dataset.num_samples()
    assert num_samples >= 0, 'Invalid number of data samples'
    assert num_samples != 0, 'Dataset has no data samples'

    # Check samples
    indices = [random.randint(0, num_samples-1) for _ in range(20)]
    indices.append(0)
    indices.append(num_samples-1)
    for index in indices:
        sample = dataset.get_sample(index)
        assert sample.shape == sample_dims, 'Unexpected dimensions for data sample'
        for node in sample:
            assert 0 <= node <= max_graph_node_id, \
                'Invalid graph node ID in data sample'
