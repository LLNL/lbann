import lbann
from lbann.modules import Module
from lbann.util import str_list

class GCN_Layer(Module):
    global_count = 0
    def __init__(self, input_channels, output_channels, name=None):
        super().__init__()
        GCN_Layer.global_count += 1

        self.name = (name if name else 'GCN_{}'.format(GCN_Layer.global_count))
        
                                
        self.weights = lbann.Weights(initializer = lbann.NormalInitializer(mean=1, standard_deviation=0),
                                    name=self.name+'_Weights')

        self.W = lbann.WeightsLayer(dims = str_list([input_channels, output_channels]),
                               name = self.name+'_layer',
                               weights =self.weights)
    def forward(self,X,A):
        B = lbann.MatMul(A,X, name=self.name+'_message')
        out = lbann.MatMul(B,self.W, name=self.name+'_aggregate')
        return out 


if __name__== '__main__':
    X = GCN_Layer(1,3)
