import lbann
from lbann.modules import Module
from lbann.util import str_list
from Graph_Data import lbann_Data_Mat
import lbann.modules.base

class Dense_GCN_Layer(Module):
    global_count = 0
    def __init__(self, input_channels, output_channels, name=None):
        super().__init__()
        Dense_GCN_Layer.global_count += 1

        self.name = (name if name else 'Dense_GCN_{}'.format(Dense_GCN_Layer.global_count))
        
                                
        self.weights = lbann.Weights(initializer = lbann.NormalInitializer(mean=0, standard_deviation=1/output_channels),
                                    name=self.name+'_Weights')

        self.W = lbann.WeightsLayer(dims = str_list([input_channels, output_channels]),
                               name = self.name+'_layer',
                               weights =self.weights)
    def forward(self,X,A):
        out = lbann.MatMul(X,self.W, name=self.name+'_aggregate')
        out = lbann.MatMul(A, out, name=self.name+'_message')
        return out 

class GCN_Layer(Module):
    global_count = 0

    def __init__(self, input_channels, output_channels, name=None, activation = None):
        super().__init__()
        GCN_Layer.global_count +=1
        self.name = (self.name if name else 'GCN_{}'.format(GCN_Layer.global_count))
        self.weights = lbann.Weights(initializer = lbann.NormalInitializer(mean = 0, standard_deviation=1/output_channels),
                                    name = self.name+'_Weights')
        self.W = lbann.WeightsLayer(dims = str_list([input_channels, output_channels]),
                                    name = self.name+'_layer',
                                    weights = self.weights)
        self.input_channels = input_channels
        self.output_channels = output_channels
    def forward(self, X, A, activation = 'relu'):
        # Assume X is a lbann data object
        for i in range(X.shape[0]):
            X[i] = lbann.MatMul(X[i], self.W, name=self.name+'_message_'+str(i))
        out = X.get_mat()
        out = lbann.MatMul(A, out, name=self.name+'_aggregate')

        #
        # Need to handle conversion back to Data
        #
        out = lbann.Relu(out, name = self.name+'_activation')
        out = lbann_Data_Mat.mat_to_data(out, X.shape[0], self.output_channels)
        return out 
            

class GIN_Layer(Module):
    global_count = 0 
    def __init__(self, nn, eps = 1e-6, name=None, activation = None):
        super().__init__()
        GIN_Layer.global_count += 1 

        self.name = (name if name else 'GIN_{}'.format(GIN_Layer.global_count))
        self.eps = eps
        fc = lbann.modules.FullyConnectedModule
        self.fc2 = fc(128)
        self.fc1 = fc(1024)
    def forward(self, X, A, activation='relu'): 
        in_channel = X.shape[1]
        eps = lbann.Constant(value=self.eps, num_neurons = str(in_channel))
        ##############################################################################
        #
        # Propagate Phase: To Do: Encase this is in a method of message and aggregate
        #
        ###############################################################################


        
        for i in range(X.shape[0]):
            temp = self.fc2(X[i]) 
            temp = lbann.Relu(temp)
            temp = self.fc1(temp)
            X[i] = temp
        out = X.get_mat()
        ###############################################################################

        out = lbann.MatMul(A,out)

        #
        # Need to handle conversion back to LBANN Data Type here
        #
        out = lbann_Data_Mat.mat_to_data(out, X.shape[0], 128)
        return out 


class Dense_Graph_Layer(Module):
    global_count = 0 
    def __init__(self, input_channels, output_channels, name=None):
        super().__init__()
        self.name = (name if name else 'Dense_Graph_{}'.format(Dense_Graph_Layer.global_count))
        
        Dense_Graph_Layer.global_count+=1                        
        self.weights_1 = lbann.Weights(initializer = lbann.NormalInitializer(mean=1, standard_deviation=0),
                                    name=self.name+'_Weights_1')
        self.weights_2 = lbann.Weights(initializer = lbann.NormalInitializer(mean=1, standard_deviation=0),
                                    name=self.name+'_Weights_2')
        self.W1 = lbann.WeightsLayer(dims = str_list([input_channels, output_channels]),
                                    name=self.name+'_param_1',
                                    weights = self.weights_1)
        self.W2 = lbann.WeightsLayer(dims = str_list([input_channels, output_channels]),
                                    name=self.name+'_param_2',
                                    weights = self.weights_2)
    def forward(self, X, A):
        messages = lbann.MatMul(X, self.W2, name=self.name+'_aggr')
        messages = lbann.MatMul(A,messages,name=self.name+'_message')

        ident = lbann.MatMul(X, self.W1, name=self.name+'_inner')

        out = lbann.Sum(ident, messages)

        return out 
        
if __name__== '__main__':
    X = GCN_Layer(1,3)
