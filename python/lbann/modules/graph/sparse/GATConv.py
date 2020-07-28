import lbann 
from lbann.modules import Module 
from lbann.util import str_list 
import lbann.modules

class GatedGraphConv(Module):
    global_count = 0
    def __init__(self, output_channels, num_layers):
        """Initialize GatedGraph layer
        """
        GatedGraphConv.global_count +=1

        self.weights = [] 
        for i in range(num_layers):
            self.weights[i] = lbann.Weights(initializer = lbann.NormalInitializer(mean=0, 
                                                          standard_deviation = 1/(i * output_channels)))
        self.output_channels = output_channels
        
        self.GRU = lbann.modules.GRUCell(output_channels)

        self.num_layers = num_layers

    def forward(self, X_data, A):
        """Call the GatedGraphConv
        """

        if (X_data.size(0) < self.output_channels):
            for i in range(X_data.size(0)):
                zeros = lbann.Constant(value = 0, num_neurons = str(self.output_channels - X_data.size(0)))
                X_data[i] = lbann.Concatenation(X_data[i], zeros, axis = 0)

        for i in self.num_layers: 
            for i in range(X.shape[0]):
                X[i] = lbann.MatMul(X[i], self.W[i])
            mat = X.get_mat()

            M = lbann_Data_Mat.mat_to_data(mat, X.shape[0], self.output_channels)

            #self.GRU()





