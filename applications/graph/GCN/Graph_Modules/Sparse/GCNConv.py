import lbann
from lbann.modules import Module
from lbann.util import str_list
import os.path
import sys

current_file = os.path.realpath(__file__)
root_dir = os.path.dirname(os.path.dirname(current_file))
sys.path.append(root_dir)

from .Graph_Data import lbann_Data_Mat
import lbann.modules.base
import math 

class GCNConv(Module):
    global_count = 0

    def __init__(self, input_channels, output_channels, name=None, activation = None):
        super().__init__()
        GCNConv.global_count +=1
        self.name = (self.name if name else 'GCN_{}'.format(GCN_Layer.global_count))
        
        std_dev = math.sqrt(6/ (input_channels + output_channels))

        self.weights = lbann.Weights(initializer = lbann.NormalInitializer(
                                                   mean = 0,
                                                   standard_deviation=std_dev),
                                    name = self.name+'_Weights')

        self.W = lbann.WeightsLayer(dims = str_list([input_channels, output_channels]),
                                    name = self.name+'_layer',
                                    weights = self.weights)
        
        self.input_channels = input_channels
        self.output_channels = output_channels
    
    def forward(self, X, A, activation = lbann.Relu):
        # Assume X is a lbann data object
        for i in range(X.shape[0]):
            X[i] = lbann.MatMul(X[i], self.W, name=self.name+'_message_'+str(i))
        out = X.get_mat()
        out = lbann.MatMul(A, out, name=self.name+'_aggregate')

        out = activation(out, name = self.name+'_activation')

        out = lbann_Data_Mat.mat_to_data(out, X.shape[0], self.output_channels)
        return out 
 

