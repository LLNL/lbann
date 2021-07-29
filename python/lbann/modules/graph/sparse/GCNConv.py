import lbann
from lbann.modules import Module, ChannelwiseFullyConnectedModule
from lbann.modules.graph.utils import GraphExpand, GraphReduce
from lbann.util import str_list
import lbann.modules.base
import math 

class GCNConv(Module):
    """GCN Conv later. See: 

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
        """Initialize GCN layer
        
        Args: 
            input_channels (int): The size of the input node features 
            output_channels (int): The output size of the node features 
            num_nodes (int): Number of vertices in the graph
            bias (bool): Whether to apply biases after weights transform 
            activation (type): Activation leyer for the node features. If None, then no activation is 
                                applied. (default: lbann.Relu)
            name (str): Default name of the layer is GCN_{number}
        """
        super().__init__()
        
        ## Add variables
        
        self.input_channel_size = input_channels
        self.output_channel_size = output_channels
        self.num_nodes = num_nodes

        ## Add Name for the components for the layer
        GCNConv.global_count +=1
        self.name = (name 
                     if name 
                     else 'GCN_{}'.format(GCNConv.global_count))
        
        weights = []
        ## Initialize weights for the matrix
        value  = math.sqrt(6/ (input_channels + output_channels))
        weights.append(lbann.Weights(initializer = lbann.UniformInitializer(min = -value,
                                                                            max = value),
                                                                            name = self.name+'_weights'))
        ## Initialize bias variables
        self.has_bias = bias

        if (self.has_bias):
            weights.append(lbann.Weights(initializer = lbann.ConstantInitializer(value = 0.0),
                                                                                 name = self.name+'_bias_weights'))


        self.activation = None 

        if activation:
            if isinstance(activation, type):
                self.activation = activation 
            else:
                self.activation = type(actvation)
            if not issubclass(self.activation, lbann.Layer):
                raise ValueError('activation must be a layer')  
        self.nn = \
            ChannelwiseFullyConnectedModule(self.output_channels,
                                            bias=self.has_bias,
                                            weights=weights,
                                            activation=self.activation,
                                            name=self.name+"_FC_layer")
    
    def forward(self, node_feature_mat, source_indices, target_indices):
        """Apply GCN

        Args:
            node_feature_mat (Layer): Node feature matrix with the shape of (num_nodes,input_channels) 
            source_indices (Layer): Source node indices of the edges with shape (num_nodes)
            target_indices (Layer): Target node indices of the edges with shape (num_nodes)
        Returns:     
            (Layer) : The output after kernel ops. The output can passed into another Graph Conv layer
                          directly
        """
        
        
        new_features = self.nn(node_feature_mat) # W \times node_feature_mat
        neighborhoods = GraphExpand(new_features, target_indices)
        reduced_features = GraphReduce(neighborhoods, source_indices, [self.num_nodes, self.output_channel_size])

        return reduced_features 
 

