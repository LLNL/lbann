import lbann 
from lbann.modules import Module 
from lbann.modules.graph.utils import GraphVertexData

class GINConv(Module):
    """Details of the kernel is detailed in: 
       https://arxiv.org/abs/1810.00826
    """
    global_count = 0; 

    def __init__(self, sequential_nn, output_channels, eps = 1e-6,name = None):
        """Initialize graph kernel as described in Graph Isomorphism Network.
           
           Args:sequential_nn ([Layers] or (Layers)): 
                output_channels (int):
                eps (float):
                name (str): 
                data_layout (str): 
        """
        GINConv.global_count += 1
        self.name = (self.name if name else 'GIN_{}'.format(GINConv.global_count))

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
            
            LBANN_Data_Mat: The output after GCN. The output can passed into another Graph Conv layer
                          directly
        """
        in_channel = X.shape[1]
        
        eps = lbann.Constant(value=(1+self.eps),num_neurons = str(in_channel))

        for layer in self.nn:
            for node_feature in range(X.shape[0]):
                eps_val = lbann.Multiply(eps, X[node_feature])
                X[node_feature] = layer(X[node_feature])
                X[node_feature] = lbann.Sum(X[node_feature],eps_val)

        out = X.get_mat(self.output_channels) #Gather the rows 
        

        out = lbann.MatMul(A, out, name=self.name+"_GIN_MATMUL")
        
        out = activation(out)
        return GraphVertexData.matrix_to_graph(out, X.shape[0], self.output_channels) 

