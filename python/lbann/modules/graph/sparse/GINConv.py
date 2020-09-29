import lbann 
from lbann.modules import Module 
from lbann.modules.graph.utils import GraphVertexData
from lbann.util import str_list

class GINConv(Module):
    """Details of the kernel is available in: 
       https://arxiv.org/abs/1810.00826
    """
    global_count = 0; 

    def __init__(self, 
                 sequential_nn,
                 output_channels,
                 eps = 1e-6,
                 name = None,
                 data_layout = 'data_parallel'):
        """Initialize graph kernel as described in Graph Isomorphism Network.
           
        Args:
            sequential_nn ([Module] or (Module)): A list or tuple of layer modules to be used  
            output_channels (int): The output size of the node features
            eps (float): Default value is 1e-6
            name (str): Default name of the layer is GIN_{number}
            data_layout (str): Data layout
        """
        GINConv.global_count += 1
        self.name = (name 
                     if name 
                     else 'GIN_{}'.format(GINConv.global_count))
        self.data_layout = data_layout
        self.nn = sequential_nn
        self.eps = eps 
        self.output_channels = output_channels


    def forward(self, X, A, activation = lbann.Relu):
        """Apply GIN  Layer. 
        
        Args:
            X (GraphVertexData): LBANN Data object, which is a collection of Layers. Each Layer is of
                                 the shape (1,input_channels) 

            A (Layer): Adjacency matrix input with shape (num_nodes, num_nodes)

            activation (Layer): Activation layer for the node features. If None, then no activation is 
                                applied. (default: lbann.Relu) 
        Returns: 
            
            (GraphVertexData): The output after GCN. The output can passed into another Graph Conv layer
                          directly
        """
        in_channel = X.shape[1]

        # Accumulate Messages from Neighboring Nodes
        out = X.get_mat()
        out = lbann.MatMul(A,out, name = self.name+"_GIN_MATMUL")
        message = GraphVertexData.matrix_to_graph(out, X.shape[0], in_channel)

        # Aggregate Messages into node features  
        eps = lbann.Constant(value=(1+self.eps),num_neurons = str_list([1, in_channel]))
        for node_feature in range(X.shape[0]):
            eps_val = lbann.Multiply(eps, X[node_feature])
            X[node_feature] = lbann.Sum(message[node_feature], eps_val)
        
        # Transform with the sequence of linear layers
        for layer in self.nn:
            for node_feature in range(X.shape[0]):
                X[node_feature] = layer(X[node_feature])
        
        ## Apply activation 
        if activation:
            for node_feature in range(X.shape[0]):
                X[node_feature] = activation(X[node_feature])
        X.update_num_features(self.output_channels) 
        return X
