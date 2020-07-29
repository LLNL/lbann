import lbann
from lbann.modules import Module
from lbann.modules.graph.utils import GraphVertexData
from lbann.util import str_list
import lbann.modules.base
import math 

class GCNConv(Module):
    """ GCN Conv later. See: 

    https://arxiv.org/abs/1609.02907
    
    """
    
    global_count = 0

    def __init__(self, 
                 input_channels,
                 output_channels,
                 bias=True,
                 name=None):
        """Initialize GCN layer
        

        Args: 
            input_channels (int): The size of the input node features 
            output_channels (int): The output size  of the node features 
            bias (bool): Whether to apply biases after MatMul 
            name (str): Default name of the layer is GCN_{number}
            data_layout (str): Data layout
            activation (type): Activation leyer for the node features

        """
        super().__init__()
        
        ## Add variables
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.data_layout = data_layout

        ## Add Name for the components for the layer
        GCNConv.global_count +=1
        self.name = (self.name if name else 'GCN_{}'.format(GCN_Layer.global_count))
        
        
        ## Initialize weights for the matrix
        std_dev = math.sqrt(6/ (input_channels + output_channels))

        self.mat_weights = lbann.Weights(initializer = lbann.UniformInitializer(
                                                   mean = 0,
                                                   standard_deviation=std_dev),
                                    name = self.name+'_Weights')

        self.W = lbann.WeightsLayer(dims = str_list([input_channels, output_channels]),
                                    name = self.name+'_layer',
                                    weights = self.mat_weights)
        
        ## Initialize bias variables
        self.has_bias = bias
        self.bias_weights = None
        self.bias = None

        if (self.has_bias):
            self.bias_weights = lbann.Weights(initializer = lbann.ConstantInitializer(
                                                            value = 0.0),
                                              name = self.name+'_bias_weights')
            self.bias = lbann.WeightsLayer(dims = str(output_channels), 
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
    
    def forward(self, X, A, activation = lbann.Relu):
        """Apply GCN

        Args:
            X (GraphVertexData): LBANN Data object, which is a collection of Layers. Each Layer is of
                                 the shape (1,input_channels) 

            A (Layer): Adjacency matrix input with shape (num_nodes, num_nodes)

                                applied. (default: lbann.Relu) 
        Returns: 
            
            LBANN_Data_Mat: The output after GCN. The output can passed into another Graph Conv layer
                          directly
        """
        
        # Assume X is a lbann data object
        for i in range(X.shape[0]):
            X[i] = lbann.MatMul(X[i], self.W, name=self.name+'_message_'+str(i))
            if (self.bias):
                X[i] = lbann.Sum(X[i], self.bias, name=self.name+'_message_bias_'+str(i))

        # Pass Message to Node Features
        out = X.get_mat()
        out = lbann.MatMul(A, out, name=self.name+'_aggregate')
        
        if self.activation:
            out = activation(out)

        out = GraphVertexData.matrix_to_graph(out, X.shape[0], self.output_channels)
        
        return out 
 

