import numpy as np
import configparser
import os.path as osp
import os


data_dir = osp.dirname(osp.realpath(__file__))
config_dir = osp.dirname(data_dir)

config = configparser.ConfigParser()
_file_name = "SIM_SGCNN_Config.ini" 

conf_file = osp.join(config_dir, _file_name)
print("Initializing using: ", conf_file)
config.read(conf_file)

NUM_SAMPLES = int(config['Graph']['num_samples'])
NUM_NODES = int(config['Graph']['num_nodes'])
NUM_COV_EDGES = int(config['Graph']['num_cov_edges'])
NUM_NON_COV_EDGES = int(config['Graph']['num_non_cov_edges'])
NUM_NODE_FEATURES = int(config['Graph']['num_node_features'])
NUM_EDGE_FEATURES = int(config['Graph']['num_edge_features'])


class Sim_GCNN_Dataset(object):
    """docstring for SimDataset"""
    def __init__(self,
                 num_samples,
                 num_nodes,
                 num_cov_edges,
                 num_non_cov_edges,   
                 node_features,
                 edge_features):
        super(Sim_GCNN_Dataset, self).__init__()
        self.num_samples = num_samples
        self.num_nodes = num_nodes
        self.node_features = node_features
        self.edge_features = edge_features
        self.num_covalent_edges = num_cov_edges
        self.num_non_covalent_edges = num_non_cov_edges
        self.data = self.__generate_data()

    def __generate_data(self):
        sample_size = self.sample_size()
        return np.zeros((self.num_samples, sample_size), dtype=np.float32)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return self.data[index].flatten()

    def sample_size(self):

        node_features_mat = self.num_nodes * self.node_features
        edge_features_mat = self.num_non_covalent_edges * self.edge_features

        covalent_edge_COO = 2 * self.num_covalent_edges
        non_covalent_edge_COO = 2 * self.num_non_covalent_edges
        ligand_only_mask = self.num_nodes

        return node_features_mat + edge_features_mat + covalent_edge_COO + non_covalent_edge_COO \
            + ligand_only_mask + 1


dataset = Sim_GCNN_Dataset(num_samples=NUM_SAMPLES, 
                           num_nodes=NUM_NODES,
                           num_cov_edges=NUM_COV_EDGES,
                           num_non_cov_edges=NUM_NON_COV_EDGES,
                           node_features=NUM_NODE_FEATURES,
                           edge_features=NUM_EDGE_FEATURES)


def get_train(index):
    return dataset[index]


def num_train_samples():
    return len(dataset)


def sample_dims():
    return (dataset.sample_size(),)


if __name__ == '__main__':
    print(len(dataset[0]))
