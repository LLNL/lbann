import lbann 
from lbann.modules import Module 
from lbann.modules.graph.utils import GraphExpand, GraphReduce
from lbann.util import str_list

class GINConv(Module):
    """Details of the kernel is available in: 
       https://arxiv.org/abs/1810.00826
    """
    global_count = 0; 

    def __init__(self, 
                 sequential_nn,
                 input_channels,
                 output_channels,
                 num_nodes,
                 eps = 1e-6,
                 name = None):
        """Initialize graph kernel as described in Graph Isomorphism Network.
           
        Args:
            sequential_nn ([Module] or (Module)): A list or tuple of layer modules to be used
            input_channels (int): The size of the input node features  
            output_channels (int): The output size of the node features
            num_nodes (int): Number of vertices in the graph
            eps (float): Default value is 1e-6
            name (str): Default name of the layer is GIN_{number}
            data_layout (str): Data layout
        """
        GINConv.global_count += 1
        self.name = (name 
                     if name 
                     else 'GIN_{}'.format(GINConv.global_count))
        self.nn = sequential_nn
        self.eps = eps 
        self.input_channel_size = input_channels
        self.output_channel_size = output_channels
        self.num_nodes = num_nodes

    def forward(self,
                node_feature_mat,
                source_indices,
                target_indices,
                activation = lbann.Relu):
        """Apply GIN  Layer. 
        
        Args:
            node_feature_mat (Layer): Node feature matrix with the shape of (num_nodes,input_channels) 
            source_indices (Layer): Source node indices of the edges with shape (num_nodes)
            target_indices (Layer): Target node indices of the edges with shape (num_nodes
            activation (Layer): Activation layer for the node features. If None, then no activation is 
                                applied. (default: lbann.Relu) 
        Returns: 
            (Layer) : The output after kernel ops. The output can passed into another Graph Conv layer
                          directly
        """

        # Accumulate Messages from Neighboring Nodes
        

        # Aggregate Messages into node features  
        eps = lbann.Constant(value=(1+self.eps),
                             num_neurons = str_list([self.num_nodes, 
                                                     self.input_channel_size]))
        
        eps_node_features = lbann.Multiply(node_feature_mat, eps, name=self.name+"_epl_mult")

        neighborhoods = GraphExpand(node_feature_mat, target_indices)
        reduced_features = GraphReduce(neighborhoods, source_indices, [self.num_nodes, 
                                                                       self.input_channel_size])


        aggregated_node_features = lbann.Sum(neighborhoods, reduced_features)
        
        # Transform with the sequence of linear layers
        for layer in self.nn:
            aggregated_node_features = layer(aggregated_node_features)

        
        ## Apply activation 
        if activation:
            aggregated_node_features = activation(aggregated_node_features)
        
        return aggregated_node_features
