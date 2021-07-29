import lbann 
from lbann.modules import Module 
from lbann.util import str_list
import lbann.modules
import math 

class GatedGraphConv(Module):
    """Gated Graph Convolution layer. For kernel details, see: 

    https://arxiv.org/abs/1511.05493

    Implementation in the spirit of:

    https://github.com/rusty1s/pytorch_geometric/blob/\
    master/torch_geometric/nn/conv/gated_graph_conv.py 
    """
    global_count = 0
    def __init__(self,
                 input_channels, 
                 output_channels,
                 num_nodes,
                 num_layers = 1,
                 name = None):
        """Initialize GatedGraph layer
        Args: 
            output_channels (int): The output size of the node features 
            num_layers (int): Number of passes through the GRU (default: 1) 
            name (str): Name of the layers and prefix to use for the layers. 
            data_layout (str): Data layout (default: data parallel)  
        """
        super().__init__()

        ## Add Name for the components for the layer 
        GatedGraphConv.global_count +=1
        self.name = (name 
                    if name 
                    else 'GatedGraphConv_{}'.format(GatedGraphConv.global_count))


        ## Add variables
        self.output_channel_size = output_channels
        self.input_channel_size = input_channels
        self.num_nodes = num_nodes

        self.rnn  = lbann.modules.ChannelwiseGRU(output_channels, num_nodes)

        self.num_layers = num_layers
        self.nns = [] 

        for i in range(num_layers):
            
            weight_init = lbann.Weights(initializer = lbann.UniformInitializer(min =-1/(math.sqrt(output_channels)), 
                                                                               max = 1/(math.sqrt(output_channels))))
            weight_layer = lbann.WeightsLayer(dims = str_list([output_channels, output_channels]),
                                              weights = weight_init, 
                                              name = self.name+'_'+str(i)+'_weight',
                                              data_layout = self.data_layout)
            self.weights.append(weight_layer)
        

    def forward(self, node_feature_mat, source_indices, target_indices):
        """Call the GatedGraphConv
        Args:
            node_feature_mat (Layer): Node feature matrix with the shape of (num_nodes,input_channels) 
            source_indices (Layer): Source node indices of the edges with shape (num_nodes)
            target_indices (Layer): Target node indices of the edges with shape (num_nodes)
        Returns:     
            (Layer) : The output after kernel ops. The output can passed into another Graph Conv layer
                          directly
        """

        if (self.input_channel_size < self.output_channel_size):
            num_zeros = self.output_channels - self.input_channel_size 
            zeros = lbann.Constant(value = 0, num_neurons = str_list([self.num_nodes,num_zeros]), name = self.name+'_padded')
            node_feature_mat = lbann.Concatenation(node_feature_mat, zeros, axis = 1)       
            
        elif (input_features > self.output_channels):
            ValueError('The feature size of the nodes {} cannot be greater than the output dimension {}'.
                        format(input_features, self.output_channels))

        for layer in range(self.num_layers): 
        
            messages = self.nns(node_feature_mat) 
            neighborhoods = GraphExpand(neighborhoods, target_indices)
            aggregate = GraphReduce(neighborhoods,source_indices, [self.num_nodes, self.output_channel_size])
            node_feature_mat = self.rnn(aggregate, node_feature_mat)
        
        return node_feature_mat
