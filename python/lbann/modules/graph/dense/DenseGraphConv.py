import lbann
from lbann.modules import Module 
from lbann.util import str_list
import math

class DenseGraphConv(Module):
    global_count = 0 
    def __init__(self, input_channels, output_channels, name=None):
        super().__init__()
        self.name = (name if name else 'DenseGraph_{}'.format(DenseGraphConv.global_count))
        
        DenseGraphConv.global_count+=1                        
        
        bounds = math.sqrt(6.0/(input_channels + output_channels))

        self.weights_1 = lbann.Weights(initializer = lbann.UniformInitializer(min=-bounds, max=bounds),
                                    name=self.name+'_Weights_1')
        self.weights_2 = lbann.Weights(initializer = lbann.UniformInitializer(min=-bounds, max=bounds),
                                    name=self.name+'_Weights_2')
        self.W1 = lbann.WeightsLayer(dims = str_list([input_channels, output_channels]),
                                    name=self.name+'_param_1',
                                    weights = self.weights_1)
        self.W2 = lbann.WeightsLayer(dims = str_list([input_channels, output_channels]),
                                    name=self.name+'_param_2',
                                    weights = self.weights_2)
    def forward(self, X, A):
        messages = lbann.MatMul(X, self.W2, name=self.name+'_w2_mult')
        messages = lbann.MatMul(A,messages,name=self.name+'_adj_mult')

        ident = lbann.MatMul(X, self.W1, name=self.name+'_w1_mult')

        out = lbann.Sum(ident, messages, name=self.name+'_sum_id')

        return out
