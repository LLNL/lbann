import os 
import os.path

import numpy as np 
from torch.utils.data import Dataset 
import utils

data_dir = os.path.dirname(os.path.realpath(__file__))


class MNIST_Superpixel_Dataset(Dataset):
    def __init__(self, train=True):
        super(MNIST_Superpixel_Dataset, self).__init__()
        self.num_vertices = 75 #All graphs have 75 nodes 
        if (train):
            self.num_data = 60000
        else:
            self.num_data = 20000

        self.node_features, self.positions, self.edges,  self.targets = self.load_processed_training()
    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        data_x = self.node_features[idx].flatten()
        data_edges = self.edges[idx].flatten()
        data_target = self.targets[idx].flatten()
        
        return np.concatenate([data_x, data_edges, data_target])

    def load_processed_training(self):

        try:
        # To Do: Add check to see if processed files exist or fall back to process_data
            node_feature_file = os.path.join(data_dir, "node_features.npy")
            positions_file    = os.path.join(data_dir, "positions.npy")
            adj_mat_file      = os.path.join(data_dir, "adj_matrices.npy")
            targets_file      = os.path.join(data_dir, "targets.npy")
        
            node_features = np.load(node_feature_file)
            positions = np.load(positions_file)
            adj_matrices = np.load(adj_mat_file)
            targets = np.load(targets_file)

            return node_features, positions, adj_matrices, targets 
        except FileNotFoundError:
            print("File not found")
            utils.download_data()
            utils.process_training_data()
            return self.load_processed_training()
            

training_data = MNIST_Superpixel_Dataset(train=True)

def get_train(index):
    return training_data[index]

def num_train_samples():
    return len(training_data)

def sample_dims():
    adjacency_matrix_size = 75 * 75 
    node_feature_size = 75 
    target_size = 10
    return (adjacency_matrix_size + node_feature_size + target_size, )

if __name__ == '__main__':
    dataset = MNIST_Superpixel_Dataset(processed=True)
