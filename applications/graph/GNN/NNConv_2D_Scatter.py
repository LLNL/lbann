import lbann 
from lbann.util import str_list, make_iterable 

import lbann
from lbann.modules import Module
from lbann.util import str_list
import math 

class ChannelwiseFullyConnectedModule(Module):
  """Basic block for channelwise fully-connected neural networks.

    Applies a dense linearity channelwise and a nonlinear activation function.
  
  """

  global_count = 0

  def __init__(self,
               size,
               bias=False,
               weights=[],
               activation=None,
               transpose=False,
               name=None, 
               parallel_strategy={}):
    """Initalize channelwise fully connected module

    Args:
        size (int): Size of output tensor.
        bias (bool): Whether to apply bias after linearity.
        transpose (bool): Whether to apply transpose of weights
                matrix.
        weights (`Weights` or iterator of `Weights`): Weights in
                fully-connected layer. There are at most two: the
                matrix and the bias. If weights are not provided, the
                matrix will be initialized with He normal
                initialization and the bias with zeros.
        activation (type): Layer class for activation function.
        name (str): Default name is in the form 'fcmodule<index>'.
        data_layout (str): Data layout.
        parallel_strategy (dict): Data partitioning scheme.
    """
    super().__init__()
    ChannelwiseFullyConnectedModule.global_count += 1
    self.instance = 0
    self.size = size
    self.bias = bias
    self.transpose = transpose
    self.name = (name
                 if name
                 else 'channelwisefc{0}'.format(ChannelwiseFullyConnectedModule.global_count))
    self.data_layout = 'data_parallel'
    self.parallel_strategy = parallel_strategy

    self.weights = list(make_iterable(weights))
    if len(self.weights) > 2:
        raise ValueError('`FullyConnectedModule` has '
                         'at most two weights, '
                         'but got {0}'.format(len(self.weights)))
    if len(self.weights) == 0:
        self.weights.append(
            lbann.Weights(initializer=lbann.HeNormalInitializer(),
                          name=self.name+'_matrix'))
    if self.bias and len(self.weights) == 1:
        self.weights.append(
            lbann.Weights(initializer=lbann.ConstantInitializer(value=0.0),
                          name=self.name+'_bias'))
    self.activation = None
    if activation:
        if isinstance(activation, type):
            self.activation = activation
        else:
            self.activation = type(activation)
        if not issubclass(self.activation, lbann.Layer):
            raise ValueError('activation must be a layer')

  def forward(self, x):
    self.instance += 1
    name = '{0}_instance{1}'.format(self.name, self.instance)
    y = lbann.ChannelwiseFullyConnected(x,
                                        weights=self.weights,
                                        name=(name+'_fc' if self.activation else name),
                                        data_layout=self.data_layout,
                                        output_channel_dims=[1, self.size],
                                        bias=self.bias,
                                        transpose=self.transpose,
                                        parallel_strategy=self.parallel_strategy)
    if self.activation:
        return self.activation(y,
                               name=name+'_activation',
                               data_layout=self.data_layout,
                               parallel_strategy=self.parallel_strategy)
    else:
        return y

class NNConv(Module):
    """Details of the kernel is available at:
       "Neural Message Passing for Quantum Chemistry"
       https://arxiv.org/abs/1704.01212
    """
    global_count = 0

    def __init__(self,
                 sequential_nn,
                 num_nodes,
                 num_edges,
                 input_channels,
                 output_channels,
                 edge_channels,
                 activation=lbann.Relu,
                 name=None):
        """Inititalize  the edge conditioned graph kernel with edge data
           represented with pseudo-COO format. The reduction over edge
           features are performed via the scatter layer
           The update function of the kernel is:
           ..  math::
                X^{\prime}_{i} = \Theta x_i + \sum_{j \in \mathcal{N(i)}}x_j \cdot h_{\Theta}(e_{i,j})
           where :math:`h_{\mathbf{\Theta}}` denotes a channel-wise NN module
        Args:
            sequential_nn ([Module] or (Module)): A list or tuple of layer
                                                  modules for updating the
                                                  edge feature matrix
            num_nodes (int): Number of vertices of each graph
                            (max number in the batch padded by 0)
            num_edges (int): Number of edges of each graph
                            (max in the batch padded by 0)
            output_channels (int): The output size of each node feature after
                                transformed with learnable weights
            activation (type): The activation function of the node features
            name (str): Default name of the layer is NN_{number}
        """
        NNConv.global_count += 1

        self.name = (name
                     if name
                     else 'NNConv_{}'.format(NNConv.global_count))

        self.output_channels = output_channels
        self.input_channels = input_channels
        self.edge_input_channels = edge_channels
        self.num_nodes = num_nodes
        self.num_edges = num_edges

        self.node_activation = activation

        self.node_nn = \
            ChannelwiseFullyConnectedModule(self.output_channels,
                                            bias=False,
                                            activation=self.node_activation,
                                            name=self.name+"_node_weights",
                                            parallel_strategy=create_parallel_strategy(2))
        self.edge_nn = sequential_nn

    def message(self,
                node_features,
                neighbor_features,
                edge_features):
        """Update node features and edge features. The Message stage of the
           convolution.
        Args:
            node_features (Layer); A 2D layer of node features of
                                   shape (num_nodes, input_channels)
            neighbor_features (Layer): A 3D layer of node features of
                                       shape (num_edges, 1, input_channels)
            edge_features (Layer): A 2D layer of edge features of
                                   shape (num_edges, edge_features)
        Returns:
            (Layer, Layer): Returns the updated node features and the messages
            for each node.
        """


        node_features = lbann.Reshape(node_features, dims=str_list([self.num_nodes, 1, self.input_channels]))
        edge_features = lbann.Reshape(edge_features, dims=str_list([self.num_edges, 1, self.edge_input_channels]))

        updated_node_features = self.node_nn(node_features)

        edge_update = None
        for layer in self.edge_nn:

            if edge_update:
                edge_update = layer(edge_update)
            else:
                edge_update = layer(edge_features)

        edge_values = \
            lbann.Reshape(edge_update,
                          dims=str_list([self.num_edges,
                                         self.input_channels,
                                         self.output_channels]),
                          name=self.name+"_edge_mat_reshape")
        edge_values = \
            lbann.MatMul(neighbor_features, edge_values)
        updated_node_features =lbann.Reshape(updated_node_features, dims=str_list([self.num_nodes, self.output_channels]))
        return updated_node_features, edge_values

    def aggregate(self,
                  edge_values,
                  edge_indices):
        """Aggregate the messages from the neighbors of the nodes
        Args:
            edge_values (Layer): A 2D layer of edge features of
                                   shape (num_edges, edge_features)
            edge_indices (Layer): A 1D layer of node features of
                                shape (num_edges * output_channels).
                                The indices used for reduction
        Returns:
            (Layer): A 2D layer of updated node features
        """

        node_feature_size = self.num_nodes * self.output_channels
        edge_feature_size = [self.num_edges , self.output_channels]

        edge_values = lbann.Reshape(edge_values,
                                    dims=str_list(edge_feature_size),
                                    name=self.name+"_neighbor_features")
        edge_reduce = lbann.Scatter(edge_values,
                                    edge_indices,
                                    dims=str_list([self.num_nodes,
                                                   self.output_channels]),
                                    axis=0,
                                    name=self.name+"_aggregate")
        edge_reduce = lbann.Reshape(edge_reduce,
                                    dims=str_list([self.num_nodes,
                                                   self.output_channels]))
        return edge_reduce

    def forward(self,
                node_features,
                neighbor_features,
                edge_features,
                edge_index):
        """Apply NNConv layer.
        Args:
            node_features (Layer): A 2D layer of node features of
                                   shape (num_nodes, input_channels)
            neighbor_features (Layer): A 3D layer of node features of
                                       shape (num_edges, 1, input_channels)
            edge_features (Layer): A 2D layer of edge features of
                                   shape (num_edges, edge_features)
            edge_index (Layer): A 1D layer of node features of
                                shape (num_edges * output_channels).
                                The indices used for reduction
        Returns:
            (Layer): The output after NNConv. The output layer has the shape
                     (num_nodes, self.output_channels)
        """

        updated_node_fts, neighbor_vals = self.message(node_features,
                                                       neighbor_features,
                                                       edge_features)
        aggregated_fts = self.aggregate(neighbor_vals, edge_index)

        update = lbann.Sum(updated_node_fts,
                           aggregated_fts,
                           name=self.name+"_updated_node_features")

        return update

def _xavier_uniform_init(fan_in, fan_out):
  """ Xavier uniform initializer

  Args:
    fan_in (int): input size of the learning layer
    fan_out (int): output size of the learning layer
  Returns:
    (UniformInitializer): return an lbann UniformInitializer object
    """
  a = math.sqrt(6 / (fan_in + fan_out))
  return lbann.UniformInitializer(min=-a, max=a)


def BondEncoder(edge_feature_columns,
      EDGE_EMBEDDING_DIM):
  """Embeds the edge features into a vector
  Args:
    edge_feature_columns (list(Layers)): A list of layers with edge feaures with shape (NUM_EDGES)
    EDGE_EMBEDDING_DIM (int): The embedding dimensionality of the edge feature vector
  Returns:
    (Layer): A layer containing the embedded edge feature matrix of shape (NUM_EDGES, EDGE_EMBEDDING_DIM)
    """
    # Courtesy of OGB
  bond_feature_dims = [5, 6, 2]
  _fan_in = bond_feature_dims[0]
  _fan_out = EDGE_EMBEDDING_DIM
  _embedding_weights = lbann.Weights(initializer=_xavier_uniform_init(_fan_in, _fan_out),
    name="bond_encoder_weights_{}".format(0))

  temp = lbann.Embedding(edge_feature_columns[0],
    num_embeddings=bond_feature_dims[0],
    embedding_dim=EDGE_EMBEDDING_DIM,
    weights=_embedding_weights,
    name="Bond_Embedding_0")

  for i in range(1, 3):
    _fan_in = bond_feature_dims[i]
    _fan_out = EDGE_EMBEDDING_DIM
    _embedding_weights = lbann.Weights(initializer=_xavier_uniform_init(_fan_in, _fan_out),
      name="bond_encoder_weights_{}".format(i))
    _temp2 = lbann.Embedding(edge_feature_columns[i],
      num_embeddings=bond_feature_dims[i],
      embedding_dim=EDGE_EMBEDDING_DIM,
      weights=_embedding_weights,
      name="Bond_Embedding_{}".format(i))
    temp = lbann.Sum(temp, _temp2)
  return temp


def AtomEncoder(node_feature_columns,
        EMBEDDING_DIM):
  """Embeds the node features into a vector

  Args:
    edge_feature_columns (list(Layers)): A list of layers with node feaures with shape (NUM_NODES)
    EMBEDDING_DIM (int): The embedding dimensionality of the node feature vector
  Returns:
    (Layer): A layer containing the embedded node feature matrix of shape (NUM_NODES, EMBEDDING_DIM)
    """
    # Courtesy of OGB
  atom_feature_dims = [119, 4, 12, 12, 10, 6, 6, 2, 2]

  _fan_in = atom_feature_dims[0]
  _fan_out = EMBEDDING_DIM

  _embedding_weights = lbann.Weights(initializer=_xavier_uniform_init(_fan_in, _fan_out),
    name="atom_encoder_weights_{}".format(0))

  temp = lbann.Embedding(node_feature_columns[0],
    num_embeddings=atom_feature_dims[0],
    embedding_dim=EMBEDDING_DIM,
    weights=_embedding_weights,
    name="Atom_Embedding_0")
  for i in range(1, 9):
    _fan_in = atom_feature_dims[i]
    _fan_out = EMBEDDING_DIM
    _embedding_weights = lbann.Weights(initializer=_xavier_uniform_init(_fan_in, _fan_out),
      name="atom_encoder_weights_{}".format(i))
    _temp2 = lbann.Embedding(node_feature_columns[i],
      num_embeddings=atom_feature_dims[i],
      embedding_dim=EMBEDDING_DIM,
      weights=_embedding_weights,
      name="Atom_Embedding_{}".format(i))
    temp = lbann.Sum(temp, _temp2)
  return temp


def graph_data_splitter(_input,
                        NUM_NODES,
                        NUM_EDGES,
                        NUM_NODE_FEATURES,
                        NUM_EDGE_FEATURES,
                        EMBEDDING_DIM,
                        EDGE_EMBEDDING_DIM):
  split_indices = []

  start_index = 0
  split_indices.append(start_index)

  node_feature = [NUM_NODES for i in range(1, NUM_NODE_FEATURES + 1)]

  split_indices.extend(node_feature)

  edge_features = [NUM_EDGES for i in range(1, NUM_EDGE_FEATURES + 1)]

  split_indices.extend(edge_features)

  edge_indices_sources = NUM_EDGES
  split_indices.append(edge_indices_sources)

  edge_indices_targets = NUM_EDGES
  split_indices.append(edge_indices_targets)

  target = 1
  split_indices.append(target)

  for i in range(1, len(split_indices)):
    split_indices[i] = split_indices[i] + split_indices[i - 1]

  graph_input = lbann.Slice(_input, axis=0,
    slice_points=str_list(split_indices))

  neighbor_feature_dims = str_list([NUM_EDGES, 1, EMBEDDING_DIM])

  node_feature_columns = [lbann.Reshape(lbann.Identity(graph_input),
    dims=str_list([NUM_NODES]),
    name="node_ft_{}_col".format(x)) for x in range(NUM_NODE_FEATURES)]

  edge_feature_columns = [lbann.Reshape(lbann.Identity(graph_input),
    dims=str_list([NUM_EDGES]),
    name="edge_ft_{}_col".format(x)) for x in range(NUM_EDGE_FEATURES)]

  source_nodes = lbann.Reshape(lbann.Identity(graph_input),
    dims=str_list([NUM_EDGES]),
    name="source_nodes")
  target_nodes = lbann.Reshape(lbann.Identity(graph_input),
    dims=str_list([NUM_EDGES]),
    name="target_nodes")
  label = lbann.Reshape(lbann.Identity(graph_input),
    dims=str_list([1]),
    name="Graph_Label")

  embedded_node_features = AtomEncoder(node_feature_columns, EMBEDDING_DIM)

  embedded_edge_features = BondEncoder(edge_feature_columns, EDGE_EMBEDDING_DIM)

  neighbor_features = lbann.Gather(embedded_node_features,
                                   target_nodes,
                                   axis=0)
  neighbor_feature_mat = lbann.Reshape(neighbor_features,
    dims=neighbor_feature_dims)
  return \
  embedded_node_features, neighbor_feature_mat, embedded_edge_features, source_nodes, label

def create_parallel_strategy(num_channel_groups):
    return {"channel_groups": num_channel_groups,
            "filter_groups": num_channel_groups}

def NNConvLayer(node_features,
        neighbor_features,
        edge_features,
        edge_index,
        in_channel,
        out_channel,
        edge_embedding_dim,
        NUM_NODES,
        NUM_EDGES):
  """ Helper function to create a NNConvLayer with a 3-layer MLP kernel

  Args: node_features (Layer): Layer containing the node featue matrix of the graph (NUM_NODES, in_channel)
        neighbor_features (Layer): Layer containing the neighbor feature tensor of the graph of shape (NUM_EDGES, 1, in_channel)
        edge_features (Layer): Layer containing the edge feature matrix of the graph of shape (NUM_EDGES, EMBEDDED_EDGE_FEATURES)
        edge_index (Layer): Layer contain the source edge index vector of the graph of shape (NUM_EDGES) 
        in_channel (int): The embedding dimensionality of the node feature vector
        out_channel (int): The dimensionality of the node feature vectors after graph convolutions
        NUM_NODES (int): The number of nodes in the largest graph in the dataset (51 for LSC-PPQM4M)
        NUM_EDGES (int): The number of edges in the largest graph in the dataset (118 for LSC-PPQM4M) 
        """
  FC = ChannelwiseFullyConnectedModule

  k_1 = math.sqrt(1 / in_channel)
  k_2 = math.sqrt(1 / 64)
  k_3 = math.sqrt(1 / 32)
  nn_sq_1_weight = lbann.Weights(initializer=lbann.UniformInitializer(min=-k_1, max=k_1),
    name="gnn_weights_{}".format(0))
  nn_sq_2_weight = lbann.Weights(initializer=lbann.UniformInitializer(min=-k_2, max=k_2),
    name="gnn_weights_weights_{}".format(1))
  nn_sq_3_weight = lbann.Weights(initializer=lbann.UniformInitializer(min=-k_3, max=k_3),
    name="gnn_weights_weights_{}".format(2))

  sequential_nn = \
  [FC(64, weights=[nn_sq_1_weight], name="NN_SQ_1", bias=True,parallel_strategy=create_parallel_strategy(2), activation=lbann.Relu),
  FC(32, weights=[nn_sq_2_weight], name="NN_SQ_2", bias=True ,parallel_strategy=create_parallel_strategy(2), activation=lbann.Relu),
  FC(out_channel * in_channel, weights=[nn_sq_3_weight], name="NN_SQ_3", bias=True,parallel_strategy=create_parallel_strategy(2))]

  nn_conv = NNConv(sequential_nn,
    NUM_NODES,
    NUM_EDGES,
    in_channel,
    out_channel,
    edge_embedding_dim)

  out = nn_conv(node_features,
    neighbor_features,
    edge_features,
    edge_index)
  return out

def make_model(NUM_NODES,
         NUM_EDGES,
         NUM_NODES_FEATURES,
         NUM_EDGE_FEATURES,
         EMBEDDING_DIM,
         EDGE_EMBEDDING_DIM,
         NUM_OUT_FEATURES,
         NUM_EPOCHS):
  """ Creates an LBANN model for the OGB-LSC PPQM4M Dataset

  Args: 
    NUM_NODES (int): The number of nodes in the largest graph in the dataset (51 for LSC-PPQM4M)
        NUM_EDGES (int): The number of edges in the largest graph in the dataset (118 for LSC-PPQM4M)
        NUM_NODES_FEATURES (int): The dimensionality of the input node features vector (9 for LSC-PPQM4M)
        NUM_EDGE_FEATURES (int): The dimensionality of the input edge feature vectors (3 for LSC-PPQM4M)
        EMBEDDOIN_DIM (int): The embedding dimensionality of the node feature vector
        EDGE_EMBEDDING_DIM (int): The embedding dimensionality of the edge feature vector
        NUM_OUT_FEATURES (int): The dimensionality of the node feature vectors after graph convolutions 
        NUM_EPOCHS (int): The number of epochs to train the network 
  Returns:
    (Model): lbann model object
    """
  in_channel = EMBEDDING_DIM
  out_channel = NUM_OUT_FEATURES
  output_dimension = 1

  _input = lbann.Input(target_mode='N/A')
  node_feature_mat, neighbor_feature_mat, edge_feature_mat, edge_indices, target = \
    graph_data_splitter(_input,
              NUM_NODES,
              NUM_EDGES,
              NUM_NODES_FEATURES,
              NUM_EDGE_FEATURES,
              EMBEDDING_DIM,
              EDGE_EMBEDDING_DIM)

  x = NNConvLayer(node_feature_mat,
          neighbor_feature_mat,
          edge_feature_mat,
          edge_indices,
          in_channel,
          out_channel,
          EDGE_EMBEDDING_DIM,
          NUM_NODES,
          NUM_EDGES)

  for i, num_neurons in enumerate([256, 128, 32, 8], 1):
    x = lbann.FullyConnected(x,
                num_neurons=num_neurons,
                name="hidden_layer_{}".format(i))
    x = lbann.Relu(x, name='hidden_layer_{}_activation'.format(i))
  x = lbann.FullyConnected(x,
    num_neurons=output_dimension,
    name="output")

  loss = lbann.MeanAbsoluteError(x, target)

  layers = lbann.traverse_layer_graph(_input)
  training_output = lbann.CallbackPrint(interval=1,
    print_global_stat_only=False)
  gpu_usage = lbann.CallbackGPUMemoryUsage()
  timer = lbann.CallbackTimer()

  callbacks = [training_output, gpu_usage, timer]
  model = lbann.Model(NUM_EPOCHS,
    layers=layers,
    objective_function=loss,
    callbacks=callbacks)
  return model