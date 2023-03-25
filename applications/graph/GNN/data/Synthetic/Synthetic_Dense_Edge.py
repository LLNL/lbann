import numpy as np
import pickle


class Synthetic_Dense_Edge(object):
  """docstring for Synthetic_Dense_Edge"""
  def __init__(self,
               num_samples,
               num_nodes,
               node_features,
               edge_features,
               use_cached=True,
               cache_data=True,
               cached_file=None):

    super(Synthetic_Dense_Edge, self).__init__()
    self.num_samples = num_samples
    self.num_nodes = num_nodes
    self.node_features = node_features
    self.edge_features = edge_features
    self.num_edges = num_nodes **2
    self.cache_data = cache_data
    node_features_size = self.num_nodes * self.node_features
    node_tensor_size = self.num_edges * self.node_features
    edge_tensor_size = self.num_edges * self.edge_features
    adj_tensor_sie = self.num_nodes * self.num_nodes
    target_size = 1

    self.sample_dim = node_features_size + node_tensor_size + edge_tensor_size + \
        adj_tensor_sie + target_size

    self.dataset = None

    if (use_cached):
      print("Using cached data")
      if (cached_file):
        self.dataset = np.load(cached_file)
      else:
        file_string = "/p/vast1/zaman2/synth_dense_graphs_{}_{}_{}_{}.p".format(num_samples,
                                                                   num_nodes,
                                                                   node_features,
                                                                   edge_features)
        try:
          with open(file_string, 'rb') as f:
            self.dataset = pickle.load(f)
        except IOError:
          print("File not found. Generating dataset")
          self.generate_data()
    else:
      self.generate_data()

  def generate_data(self):
    self.dataset = np.random.random((self.num_samples, self.sample_dim))
    _file_string = "/p/vast1/zaman2/synth_dense_graphs_{}_{}_{}_{}.p".format(self.num_samples,
                                                                  self.num_nodes,
                                                                  self.node_features,
                                                                  self.edge_features)
    with open(_file_string, 'wb') as f:
        pickle.dump(self.dataset, f, protocol=pickle.HIGHEST_PROTOCOL)


  def get_sample(self, i):
    return self.dataset[i]


number_samples = 10000
number_nodes = 100
number_node_features = 10
number_edge_features = 1


dataset = Synthetic_Dense_Edge(number_samples,
                               number_nodes,
                               number_node_features,
                               number_edge_features)


def get_sample_func(index):
  _data = dataset.get_sample(index)
  _data = np.float32(_data)
  return _data


def num_samples_func():
  return number_samples


def sample_dims_func():
  return (dataset.sample_dim,)

if __name__ == '__main__':
  print(dataset.sample_dim)