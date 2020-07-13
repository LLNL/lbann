import lbann
from lbann.util import str_list
from Graph_Kernels import GIN_Layer as GraphConv
from Graph_Data import lbann_Data_Mat, lbann_Graph_Data


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
    

    
    
    '''
    feature_matrix_size = num_vertices * node_features 
   
    graph_input = lbann.Slice(input_, axis = 0 , slice_points = str_list([0,feature_matrix_size,graph_dims, sample_dims]), name = "Graph_Input") 
    #graph = lbann.Identity(graph_input, name="Flat_Graph_Data")
    
    # Slice graph into node feature matrix, and adjacency matrix 

    #reshape 

    feature_matrix = lbann.Reshape(graph_input, dims = str_list([num_vertices, node_features]), name="Node_features")
    adj_matrix = lbann.Reshape(graph_input, dims = str_list([num_vertices,num_vertices]), name="Adj_Mat") 

    target = lbann.Identity(graph_input, name="Target")
    target = lbann.Reshape(target, dims="10")   
    '''
    #----------------------------------
    # Perform Graph Convolution
    #----------------------------------
    
    out_channel_1 = 1024
    out_channel_2 = 256
    out_channel_3 = 128
    
    gcn = GraphConv(nn = 2)
    #gcn2 = GraphConv(input_channels = out_channel_1, output_channels = out_channel_2)
    #gcn3 = 
    
    out_channel = out_channel_3 
    
    x = gcn(feature_matrix, adj_matrix, activation='relu' )
    #x = lbann.Relu(x,name="GCN1_activation") 

    #x = gcn2(x, adj_matrix, activation = 'relu')
    #x = lbann.Relu(x, name="GCN2_activation")
    
    #x = gcn3 (x, adj_matrix, activation = 'relu')
    #x = lbann.Relu(x, name="GCN3_activation")

    #----------------------------------
    # Apply Reduction on Node Features
    #----------------------------------

    average_vector = lbann.Constant(value = 1/num_vertices, num_neurons = str_list([1,num_vertices]), name="Average_Vector")
    x = x.get_mat()
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
 
