import lbann
from lbann.modules import Module
from lbann.modules.graph.utils import GraphVertexData
from lbann.util import str_list
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
                 bias=True,
                 activation = lbann.Relu,
                 name=None,
                 data_layout = 'data_parallel'):
        """Initialize Graph layer

        Args: 
            input_channels (int): The size of the input node features 
            output_channels (int): The output size  of the node features 
            bias (bool): Whether to apply biases after MatMul 
            name (str): Default name of the layer is GCN_{number}
            data_layout (str): Data layout
            activation (type): Activation layer for the node features. If None, then no activation is 
                                applied. (default: lbann.Relu)
        """
        super().__init__()
        
        ## Add variables
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.data_layout = data_layout

        ## Add Name for the components for the layer
        GraphConv.global_count +=1
        self.name = (name 
                     if name 
                     else 'Graph_{}'.format(GraphConv.global_count))
        
        ## Initialize weights for the matrix
        value  = math.sqrt(6/ (input_channels + output_channels))

        self.mat_weights = lbann.Weights(initializer = lbann.UniformInitializer(
                                                       min = -value,
                                                       max = value),
                                         name = self.name+'_Weights')

        self.weights1 = lbann.WeightsLayer(dims = str_list([input_channels, output_channels]),
                                    name = self.name+'_layer',
                                    weights = self.mat_weights)

        self.id_weights = lbann.Weights(initializer = lbann.UniformInitializer(
                                                      min  = -value,
                                                      max = value),
                                         name = self.name+'_ID_Weights')

        self.weights2 = lbann.WeightsLayer(dims = str_list([input_channels, output_channels]),
                                    name = self.name+'_ID_layer',
                                    weights = self.id_weights) 
        
        ## Initialize bias variables
        self.has_bias = bias
        self.bias_weights = None
        self.bias = None

        if (self.has_bias):
            self.bias_weights = lbann.Weights(initializer = lbann.ConstantInitializer(
                                                            value = 0.0),
                                              name = self.name+'_bias_weights')
            self.bias = lbann.WeightsLayer(dims = str_list([1,output_channels]), 
                                           weights = self.bias_weights, 
                                           name = self.name+'_bias_layer')
        
        self.activation = None 

        if activation:
            if isinstance(activation, type):
                self.activation = activation 
            else:
                self.activation = type(actvation)
            if not issubclass(self.activation, lbann.Layer):
                raise ValueError('activation must be a layer')
    
    def forward(self, X, A):
        """Apply Graph Conv Layer to X and use A for message passing

        Args:
            X (GraphVertexData): LBANN Data object, which is a collection of Layers. Each Layer is of
                                 the shape (1,input_channels) 

            A (Layer): Adjacency matrix input with shape (num_nodes, num_nodes)

        Returns: 
            
            GraphVertexData: The output after convolution. The output can passed into another Graph Conv layer
                          directly
        """ 
        
        # Accumulate Messages from Neighboring Nodes
        out = X.get_mat()
        out = lbann.MatMul(out,self.weights1, name = self.name+"_Graph_MATMUL")
        message  = lbann.MatMul(A, out, name = self.name+"_Graph_Message")
        message = GraphVertexData.matrix_to_graph(message, X.shape[0], self.output_channels)

        # Assume X is a GraphVertexData object
        
        for node_feature in range(X.shape[0]):
            X[node_feature] = lbann.MatMul(X[node_feature], self.weights2)
        
        for node_feature in range(X.shape[0]):
            if (self.bias):
                message[node_feature] = lbann.Sum(message[node_feature], 
                                                  self.bias,
                                                  name=self.name+'_message_bias_'+str(node_feature))
            X[node_feature] = lbann.Sum(X[node_feature], message[node_feature])
 
        if self.activation:
            for node_feature in range(X.shape[0]):
                X[node_feature] = self.activation(X[node_feature])

        X.update_num_features(self.output_channels) 
        return X

