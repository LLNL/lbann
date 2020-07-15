import lbann 
import lbann.modules import Module 
import math 


class DenseGCN(Module):
    global_count = 0

    def __init__(self, input_channels, output_channels, name=None):
        super().__init__()
        Dense_GCN_Layer.global_count += 1

        self.name = (name if name else 'Dense_GCN_{}'.format(Dense_GCN_Layer.global_count))
        
        
        bounds = math.sqrt(6.0 / (input_channels + output_channels))
        self.weights = lbann.Weights(initializer = lbann.UniformInitializer(min=-bounds,max=bounds),
                                    name=self.name+'_Weights')

        self.W = lbann.WeightsLayer(dims = str_list([input_channels, output_channels]),
                               name = self.name+'_layer',
                               weights =self.weights)
    def forward(self,X,A):
        out = lbann.MatMul(X,self.W, name=self.name+'_weight_mult')
        out = lbann.MatMul(A, out, name=self.name+'_adj_mult')
        return out
