import lbann
from lbann.util import str_list
from lbann.modules.graph import GINConv as GraphConv
from lbann.modules.graph import GraphVertexData as lbann_Data_Mat
from graph_data_util  import lbann_Graph_Data 

def make_model(num_vertices = None, 
               node_features = None, 
               num_classes = None,
               dataset = None,
               num_epochs = 1):
    
    '''
    
    Construct a simple single layer GCN Model. 

    '''

    assert num_vertices != dataset #Ensure atleast one of the values is set 

    if dataset is not None:
        assert num_vertices is None

        if dataset == 'MNIST':
            num_vertices = 75
            num_classes = 10
            node_features = 1

        elif dataset == 'PROTEINS':
            num_vertices = 100
            num_classes = 2
            node_features = 3
        else:
            raise Exception("Unkown Dataset")

    assert num_vertices is not None
    assert num_classes is not None 
    assert node_features is not None 
    

    #----------------------------------
    # Reshape and Slice Input Tensor 
    #----------------------------------

    input_ = lbann.Input(target_mode = 'classification')

    # Input dimensions should be (num_vertices * node_features + num_vertices^2 + num_classes )
    
    # input should have atleast two children since the target is classification 
    
    #data = lbann.Identity(input_)

    
    data = lbann_Graph_Data(input_,num_vertices, node_features,num_classes)
    
    feature_matrix = data.x 
    adj_matrix = data.adj 
    target = data.y 
      
   
    #----------------------------------
    # Perform Graph Convolution
    #----------------------------------
    
    
    out_channel = 16
    FC = lbann.modules.FullyConnectedModule
    sequential_nn = [FC(128), lbann.Relu, FC(64), lbann.Relu, FC(32), lbann.Relu, FC(16), lbann.Relu]

    gin = GraphConv(sequential_nn , output_channels= out_channel)
    
    x = gin(feature_matrix, adj_matrix, activation = lbann.Relu)
    
    #----------------------------------
    # Apply Reduction on Node Features
    #----------------------------------

    average_vector = lbann.Constant(value = 1/num_vertices, num_neurons = str_list([1,num_vertices]), name="Average_Vector")
    x = x.get_mat(out_channel)
    x = lbann.MatMul(average_vector,x, name="Node_Feature_Reduction") # X is now a vector with output_channel dimensions 
    
    x = lbann.Reshape(x, dims= str_list([out_channel]), name="Squeeze")
    x = lbann.FullyConnected(x, num_neurons=64, name="hidden_layer_1")
    x = lbann.Relu(x, name="hidden_layer_1_activation")
    x = lbann.FullyConnected(x, num_neurons=num_classes, name="Output_Fully_Connected")
    
    #----------------------------------
    # Loss Function and Accuracy s
    #----------------------------------
    
    
    probs = lbann.Softmax(x, name="Softmax")
    loss = lbann.CrossEntropy(probs, target, name="Cross_Entropy_Loss")
    accuracy = lbann.CategoricalAccuracy(probs, target, name="Accuracy")

    layers = lbann.traverse_layer_graph(input_)
    
    print_model = lbann.CallbackPrintModelDescription() #Prints initial Model after Setup

    training_output = lbann.CallbackPrint( interval = 1,
                           print_global_stat_only = False) #Prints training progress
    gpu_usage = lbann.CallbackGPUMemoryUsage()
    
    timer = lbann.CallbackTimer()

    callbacks = [print_model, training_output, gpu_usage, timer]

    metrics = [lbann.Metric(accuracy, name='accuracy', unit="%")]

    model = lbann.Model(num_epochs, 
                       layers = layers,
                       objective_function = loss,
                       metrics = metrics, 
                       callbacks = callbacks
                       )
    return model


if __name__ == '__main__':
    model = make_model(dataset="MNIST")
 
