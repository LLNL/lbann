import numpy as np
import os


# load the dataset

data_dir = "/p/vast1/lbann/datasets/OpenGraphBenchmarks/dataset/ogbn_arxiv"

connectivity_data = np.load(data_dir + "/edges.npy")
node_data = (
    np.load(data_dir + "/node_feats.npy")
    if os.path.exists(data_dir + "/node_feats.npy")
    else np.random.rand(169343, 128)  # random node features
)

labels_data = (
    np.load(data_dir + "/labels.npy")
    if os.path.exists(data_dir + "/labels.npy")
    else np.random.randint(0, 40, 169343)  # random labels
)

num_edges = 1166243
num_nodes = 169343

assert connectivity_data.shape == (num_edges, 2)
assert node_data.shape == (num_nodes, 128)


def get_train_sample(index):
    # Return the complete node data
    return node_data.flatten() + connectivity_data.flatten() + labels_data.flatten()


def sample_dims():
    return (
        np.reduce(node_data.shape, lambda x, y: x * y)
        + np.reduce(connectivity_data.shape, lambda x, y: x * y),
    )


def num_train_samples():
    return 1
