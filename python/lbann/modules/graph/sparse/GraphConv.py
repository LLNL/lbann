import lbann
from lbann.modules import Module, ChannelwiseFullyConnectedModule
from lbann.modules.graph.utils import GraphExpand, GraphReduce
import lbann.modules.base
import math

class GraphConv(Module):
    """ Graph Conv layer. See:

    https://arxiv.org/abs/1609.02907

    """

    global_count = 0

    def __init__(self,
                 input_channels,
                 output_channels,
                 num_nodes,
                 bias=True,
                 activation = lbann.Relu,
                 name=None):
        """Initialize Graph layer

        Args:
            input_channels (int): The size of the input node features
            output_channels (int): The output size  of the node features
            num_nodes (int): Number of vertices in the graph
            bias (bool): Whether to apply biases after weights transform
            activation (type): Activation layer for the node features. If None, then no activation is
                                applied. (default: lbann.Relu)
            name (str): Default name of the layer is Graph_{number}
        """
        super().__init__()

        ## Add variables

        self.input_channel_size = input_channels
        self.output_channel_size = output_channels
        self.num_nodes = num_nodes

        ## Add Name for the components for the layer
        GraphConv.global_count +=1
        self.name = (name
                     if name
                     else 'Graph_{}'.format(GraphConv.global_count))

        ## Initialize weights for the matrix
        value  = math.sqrt(6/ (input_channels + output_channels))

        mat_weights = []
        id_weights = []

        mat_weights.append(lbann.Weights(initializer = lbann.UniformInitializer(
                                                       min = -value,
                                                       max = value),
                                         name = self.name+'_Weights'))

        id_weights.append(lbann.Weights(initializer = lbann.UniformInitializer(
                                                      min  = -value,
                                                      max = value),
                                         name = self.name+'_ID_Weights'))

        ## Initialize bias variables
        self.has_bias = bias

        if (self.has_bias):
            mat_weights.append(lbann.Weights(initializer = lbann.ConstantInitializer(
                                                            value = 0.0),
                                              name = self.name+'_bias_weights'))
        self.activation = None

        if activation:
            if isinstance(activation, type):
                self.activation = activation
            else:
                self.activation = type(actvation)
            if not issubclass(self.activation, lbann.Layer):
                raise ValueError('activation must be a layer')
        self.id_nn = \
            ChannelwiseFullyConnectedModule(self.output_channel_size,
                                            bias=False,
                                            weights=id_weights,
                                            activation=self.activation,
                                            name=self.name+"_ID_FC_layer")
        self.mat_nn = \
            ChannelwiseFullyConnectedModule(self.output_channel_size,
                                            bias=self.has_bias,
                                            weights=mat_weights,
                                            activation=self.activation,
                                            name=self.name+"_Message_FC_layer")

    def forward(self, node_feature_mat, source_indices, target_indices):
        """Apply Graph Conv Layer

        Args:
            node_feature_mat (Layer): Node feature matrix with the shape of (num_nodes,input_channels)
            source_indices (Layer): Source node indices of the edges with shape (num_nodes)
            target_indices (Layer): Target node indices of the edges with shape (num_nodes)
        Returns:
            (Layer) : The output after kernel ops. The output can passed into another Graph Conv layer
                          directly
        """


        new_self_features = self.id_nn(node_feature_mat)

        new_neighbor_features = self.mat_nn(node_feature_mat)
        # Place the new features on to neighborhoods
        neighborhoods = GraphExpand(new_neighbor_features, target_indices)
        # Accumulate Messages from Neighboring Nodes
        reduced_features = GraphReduce(neighborhoods, source_indices, [self.num_nodes, self.output_channel_size])

        out_features = lbann.Sum(new_self_features, reduced_features)
        return out_features
