import lbann
from lbann.modules import Module, ChannelwiseFullyConnectedModule
from lbann.util import str_list


class DenseNNConv(Module):
  """Details of the kernel is available at:
    "Neural Message Passing for Quantum Chemistry"
    https://arxiv.org/abs/1704.01212"""
  global_count = 0

  def __init__(self,
               sequential_nn,
               num_nodes,
               input_channels,
               output_channels,
               activation=lbann.Relu,
               name=None):

      super(DenseNNConv, self).__init__()
      self.name = (name if name
                   else 'Dense_GCN_{}'.format(DenseNNConv.global_count))
      self.output_channels = output_channels
      self.input_channels = input_channels
      self.num_nodes = num_nodes
      self.node_activation = activation
      self.node_nn = \
          ChannelwiseFullyConnectedModule(self.output_channels,
                                          bias=False,
                                          activation=self.node_activation,
                                          name=self.name + "_node_weights")
      self.edge_nn = sequential_nn

  def forward(self,
              node_features_mat,
              edge_features_tensor,
              node_features_tensor,
              adjacency_tensor):

    num_edges = self.num_nodes ** 2

    edge_ft_shape = str_list([num_edges, self.input_channels, self.output_channels])
    node_ft_tensor_shape = str_list([self.num_nodes, self.num_nodes, self.output_channels])
    node_ft_mat_shape = str_list([self.num_nodes, self.output_channels])

    transformed_edge_ft_tensor = None

    for layer in self.edge_nn:
        if transformed_edge_ft_tensor is not None:
            transformed_edge_ft_tensor = layer(transformed_edge_ft_tensor)
        else:
            transformed_edge_ft_tensor = layer(edge_features_tensor)

    transformed_edge_ft_tensor = lbann.Reshape(transformed_edge_ft_tensor,
                                               dims=edge_ft_shape,
                                               name=self.name+"_edge_ft_reshape")

    new_node_features = lbann.MatMul(node_features_tensor, transformed_edge_ft_tensor)
    new_node_features = lbann.Reshape(new_node_features,
                                      dims=node_ft_tensor_shape)

    gathered_node_features = lbann.MatMul(adjacency_tensor, new_node_features)

    new_node_features = lbann.Reshape(gathered_node_features,
                                      dims=node_ft_mat_shape)
    updated_nodes = self.node_nn(node_features_mat)

    out = lbann.Sum(new_node_features, updated_nodes)

    return out