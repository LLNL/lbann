import lbann
from lbann.modules import Module, ChannelwiseFullyConnectedModule


class MLP(Module):
  """
    Applies channelwise MLP with ReLU activation with Layer Normalization 
    with a specified number of hidden layers
  """
  global_count = 0

  def __init__(self,
               in_dim,
               out_dim,
               hidden_dim,
               hidden_layers,
               norm_type=lbann.LayerNorm,
               name=None):

    super().__init__()
    MLP.global_count += 1

    self.instance = 0
    self.in_dim = in_dim
    self.out_dim = out_dim
    self.hidden_dim = hidden_dim
    self.hidden_layers = hidden_layers

    self.name = (name if name
                 else f'MLP_{MLP.global_count}')

    self.layers = [ChannelwiseFullyConnectedModule(hidden_dim,
                                                   bias=True,
                                                   activation=lbann.Relu)]
    for i in range(hidden_layers):
      # Total number of MLPs is hidden layers + 2 (input and output)
      self.layers.append(ChannelwiseFullyConnectedModule(hidden_dim,
                                                         bias=True,
                                                         activation=lbann.Relu))

    self.layers.append(ChannelwiseFullyConnectedModule(out_dim,
                                                       bias=True,
                                                       activation=None))

    self.norm_type = None

    if norm_type:
      if isinstance(norm_type, type):
        self.norm_type = norm_type
      else:
        self.norm_type = type(norm_type)

      if not issubclass(norm_type, lbann.Layer):
        raise ValueError("Normalization must be a layer")

  def forward(self, x):
    """
      Args:
        x (Layer) : Expected shape (Batch, N, self.in_dim) 

      Returns: 
        (Layer): Expected shape (Batch, N, self.out_dim) 
    """
    self.instance += 1
    name = f"{self.name}_instance_{self.instance}"

    for layer in self.layers:
      x = layer(x)

    if self.norm_type:
      return self.norm_type(x)
    return x


class EdgeProcessor(Module):
  """ Applies MLP transform on concatenated node and edge features
  """
  global_count = 0


  def __init__(self,
               in_dim_node=128,
               in_dim_edge=128,
               hidden_dim=128,
               hidden_layers=2,
               norm_type=lbann.LayerNorm,
               name=None):
    super().__init__()
    EdgeProcessor.global_count += 1
    self.instance = 0
    self.name = (name if name
                 else f'EdgeProcessor_{EdgeProcessor.global_count}')

    self.edge_mlp = MLP(2 * in_dim_node + in_dim_edge,
                        in_dim_edge,
                        hidden_dim=hidden_dim,
                        hidden_layers=hidden_layers,
                        norm_type=norm_type,
                        name=f"{self.name}_edge_mlp")
  def forward(self,
              node_features,
              edge_features,
              source_node_indices,
              target_node_indices,):
    """
      Args:
        node_features (Layer) : Expected shape (Batch, num_nodes, self.in_dim_node)
        edge_features (Layer) : Expected shape (Batch, num_edges, self.in_dim_edge)
        source_node_indices (Layer) : Expected shape (Batch, num_edges) 
        target_node_indices (Layer) : Expected shape (Batch, num_edges)

      Returns: 
        (Layer): Expected shape (Batch, Num_edges, self.in_dim_edge) 
    """
    self.instance += 1
    source_node_features = lbann.Gather(node_features, source_node_indices, axis=0)
    target_node_features = lbann.Gather(node_features, target_node_indices, axis=0)

    x = lbann.Concatenation([source_node_features, target_node_features, edge_features],
                             axis=1,
                             name=f"{self.name}_{self.instance}_concat_features")
    x = self.edge_mlp(x)

    return lbann.Sum(edge_features, x,
                     name=f"{self.name}_{self.instance}_residual_sum")


class NodeProcessor(Module):
  """ Applies MLP transform on scatter-summed edge features and node features
  """
  global_count = 0


  def __init__(self,
               num_nodes, 
               in_dim_node=128,
               in_dim_edge=128,
               hidden_dim=128,
               hidden_layers=2,
               norm_type=lbann.LayerNorm,
               name=None):
    super().__init__()
    NodeProcessor.global_count += 1
    self.instance = 0
    self.name = (name if name
                 else f'NodeProcessor_{NodeProcessor.global_count}')
    self.num_nodes = num_nodes
    self.in_dim_edge = in_dim_edge
    self.node_mlp = MLP(in_dim_node + in_dim_edge,
                        in_dim_node,
                        hidden_dim=hidden_dim,
                        hidden_layers=hidden_layers,
                        norm_type=norm_type,
                        name=f"{self.name}_node_mlp")
  
  def forward(self,
              node_features,
              edge_features,
              target_edge_indices):
    """
      Args:
        node_features (Layer) : Expected shape (Batch, num_nodes, self.in_dim_node) 
        edge_features (Layer) : Expected shape (Batch, Num_edges, self.in_dim_edge)
        edge_indices (Layer): Expected shape (Batch, Num_edges)
      Returns: 
        (Layer): Expected shape (Batch, Num_nodes, self.in_dim_node) 
    """
    self.instance += 1
  
    edge_feature_sum = lbann.Scatter(edge_features, target_edge_indices,
                                     name=f"{self.name}_{self.instance}_scatter",
                                     dims=[self.num_nodes, self.in_dim_edge],
                                     axis=0)

    x = lbann.Concatenation([node_features, edge_feature_sum],
                             axis=1,
                             name=f"{self.name}_{self.instance}_concat_features")
    x = self.node_mlp(x)

    return lbann.Sum(node_features, x,
                     name=f"{self.name}_{self.instance}_residual_sum")


class GraphProcessor(Module):
  """ Graph processor module 
  """

  def __init__(self,
               num_nodes,
               mp_iterations=15,
               in_dim_node=128, in_dim_edge=128,
               hidden_dim_node=128, hidden_dim_edge=128,
               hidden_layers_node=2, hidden_layers_edge=2,
               norm_type=lbann.LayerNorm):
    super().__init__()

    self.blocks = []

    for _ in range(mp_iterations):
      node_processor = NodeProcessor(num_nodes=num_nodes,
                                     in_dim_node=in_dim_node,
                                     in_dim_edge=in_dim_edge,
                                     hidden_dim=hidden_dim_node,
                                     hidden_layers=hidden_layers_node,
                                     norm_type=norm_type)

      edge_processor = EdgeProcessor(in_dim_node=in_dim_node,
                                     in_dim_edge=in_dim_edge,
                                     hidden_dim=hidden_dim_edge,
                                     hidden_layers=hidden_layers_edge,
                                     norm_type=norm_type)

      self.blocks.append((node_processor, edge_processor))

  def forward(self,
              node_features,
              edge_features,
              source_node_indices,
              target_node_indices):
    """
      Args:
        node_features (Layer) : Expected shape (Batch, num_nodes, self.in_dim_node) 
        edge_features (Layer) : Expected shape (Batch, Num_edges, self.in_dim_edge)
        source_node_indices (Layer) : Expected shape (Batch, num_edges) 
        target_node_indices (Layer) : Expected shape (Batch, num_edges)
      Returns: 
        (Layer, Layer): Expected shape (Batch, Num_nodes, self.in_dim_node) and 
          (Batch, num_edges, self.in_dim_edge)
    """

    for node_processor, edge_processor in self.blocks:
      x = node_processor(node_features, edge_features, target_node_indices)
      e = edge_processor(node_features, edge_features,
                         source_node_indices,
                         target_node_indices)

      node_features = x 
      edge_features = e

    return node_features, edge_features
