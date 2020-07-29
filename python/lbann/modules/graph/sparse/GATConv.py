import lbann 
from lbann.modules import Module 
from lbann.util import str_list
from lbann.modules.graph.utils import GraphVertexData
import lbann.modules

class GatedGraphConv(Module):
    global_count = 0
    def __init__(self, 
                 output_channels,
                 num_layers,
                 name = None):
        """Initialize GatedGraph layer
        
        """
        super().__init__()

        ## Add Name for the components for the layer 
        GatedGraphConv.global_count +=1
        self.name = (self.name if name else 'GatedGraphConv_{}'.format(GatedGraphConv.global_count))


        ## Add variables
        self.output_channels = output_channels
        self.GRU = lbann.modules.GRUCell(output_channels)

        self.num_layers = num_layers
        
        self.weights = [] 

        for i in range(num_layers):
            weight_init = lbann.Weights(initializer = lbann.NormalInitializer(mean=0, 
                                                          standard_deviation = 1/(i * output_channels)))
            weight_layer = lbann.WeightsLayer(dims = str_list([output_channels, output_channels] 
            self.weights.append(()
        

    def forward(self, X, A):
        """Call the GatedGraphConv
        
        """

        input_features = X.size(1)
        if (X.size(1) < self.output_channels):
            for i in range(X.size(0)):
                zeros = lbann.Constant(value = 0, num_neurons = str(self.output_channels - X.size(0)))
                X[i] = lbann.Concatenation(X[i], zeros, axis = 0)       
        elif (X.size(1) > self.output_channels):
            ValueError('The feature size of the nodes {} cannot be greater than the output dimension {}'.
                        format(X.size(1), self.output_channels)
        
        for layer in self.num_layers: 
            
            ##
            X = X.get_mat()
            messages = lbann.MatMul(X, self.weights[layer]) 
            aggregate = lbann.MatMul(A,messages)

            M = GraphVertexData.matrix_to_graph(aggregate, X.size(0), self.output_channels)

            for i in range(X.shape[0]):
                X[i] = lbann.MatMul(X[i], self.W[layer])
                _,X[i] = self.rnn(M[i], X[i])
        return X
