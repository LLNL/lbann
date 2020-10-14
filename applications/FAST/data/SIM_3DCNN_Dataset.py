import numpy as np


class Sim_CNN_Dataset(object):
    """docstring for SimDataset"""
    def __init__(self,
                 num_samples,
                 grid_size,
                 num_features):
        super(Sim_CNN_Dataset, self).__init__()
        self.grid_size = grid_size
        self.num_features = num_features
        self.data = self.__generate_data()

    def __generate_data(self):
        sample_size = self.sample_size()
        return np.zeros((self.num_samples, sample_size), dtype=np.float32)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return self.data[index].flatten()

    def sample_size(self):
        sample_size = (self.grid_size ** 3) * self.num_features
        return sample_size + 1


dataset = Sim_CNN_Dataset(100, 48, 19)


def get_train(index):
    return dataset[index]


def num_train_samples():
    return len(dataset)


def sample_dims():
    return (dataset.sample_size(),)
