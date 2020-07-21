import lbann 
from lbann.modules import Module 
import os.path
import sys 

### Local Imports
current_file = os.path.realpath(__file__)
root_dir = os.path.dirname(os.path.dirname(current_file))
sys.path.append(root_dir)
from .Graph_Data  import lbann_Data_Mat as Matrix

class GINConv(Module):
    global_count = 0; 

    def __init__(self, sequential_nn, output_channels, eps = 1e-6,name = None):
        GINConv.global_count += 1
        self.name = (self.name if name else 'GIN_{}'.format(GINConv.global_count))

        self.nn = sequential_nn
        
        self.eps = eps
        
        self.output_channels = output_channels


    def forward(self, X, A):
        in_channel = X.shape[1]
        
        eps = lbann.Constant(value=self.eps,num_neurons = str(in_channel))

        for layer in self.nn:
            for node_feature in range(X.shape[0]):
                X[node_feature] = layer(X[node_feature])

        out = X.get_mat(self.output_channels) #Gather the rows 

        out = lbann.MatMul(A, out, name=self.name+"_GIN_MATMUL")

        return Matrix.mat_to_data(out, X.shape[0], self.output_channels) 

