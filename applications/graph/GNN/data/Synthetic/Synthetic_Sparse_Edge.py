import numpy as np
import os.path as osp
import configparser
import os

cur_dir = osp.dirname(osp.realpath(__file__))
data_dir = osp.dirname(cur_dir)

config_dir = osp.dirname(data_dir)
config = configparser.ConfigParser()
_file_name = os.environ['SYNTH_TEST_CONFIG_FILE'] 

conf_file = osp.join(config_dir, _file_name)
print("Initializing using: ", conf_file)
config.read(conf_file)

num_nodes = int(config['Graph']['num_nodes'])
max_edges = int(config['Graph']['num_edges'])

number_samples = 10000
number_node_features = 9
number_edge_features = 3


def sample_dims_func():
    node_feature_size = num_nodes * number_node_features
    edge_indices_size = max_edges * 2
    edge_features_size = max_edges * number_edge_features
    return (node_feature_size + edge_indices_size + edge_features_size + num_nodes + 1,)


dataset = np.random.randint(2, size=(number_samples, sample_dims_func()[0]))


def get_sample_func(index):
    _data = np.float32(dataset[index])
    return _data


def num_samples_func():
    return number_samples


if __name__ == '__main__':
  print(num_samples_func())
  print(sample_dims_func())
  print(get_sample_func(0).shape)