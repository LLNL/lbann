import lbann
from lbann.util import str_list
from Graph_Kernels import Dense_GCN_Layer as GraphConv
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

    '''
    data = lbann_Graph_Data(input_,75, 1,10)
    
    feature_matrix = data.x 
    adj_matrix = data.adj 
    target = data.y 
    '''

    
    sample_dims = num_vertices*node_features + (num_vertices ** 2) + num_classes
    graph_dims = num_vertices*node_features + (num_vertices ** 2)
        
    feature_matrix_size = num_vertices * node_features 
   
    graph_input = lbann.Slice(input_, axis = 0 , slice_points = str_list([0,feature_matrix_size,graph_dims, sample_dims]), name = "Graph_Input") 
    #graph = lbann.Identity(graph_input, name="Flat_Graph_Data")
    
    # Slice graph into node feature matrix, and adjacency matrix 

    #reshape 

    feature_matrix = lbann.Reshape(graph_input, dims = str_list([num_vertices, node_features]), name="Node_features")
    adj_matrix = lbann.Reshape(graph_input, dims = str_list([num_vertices,num_vertices]), name="Adj_Mat") 

    target = lbann.Identity(graph_input, name="Target")
    target = lbann.Reshape(target, dims=str(num_classes))   
    #----------------------------------
    # Perform Graph Convolution
    #----------------------------------

    # To Do: Implement lbann.GCN()
    # 
    # x = lbann.GCN(feature_matrix, adj_matrix, output_channels = N) # X is now the feature_matrix of shape num_vertices x output_channels 
    #
    
   # print("Warning: Not using GCN layer and forwaring orignal feature matrix to reduction step. Should only use this when testing dataset / data reader")
    
    out_channel_1 = 256
    out_channel_2 = 64
    out_channel_3 = 32
    
    gcn = GraphConv(input_channels = node_features, output_channels = out_channel_1)
    gcn2 = GraphConv(input_channels = out_channel_1, output_channels = out_channel_2)
    gcn3 = GraphConv(input_channels = out_channel_2, output_channels = out_channel_3)
    
    out_channel = out_channel_3
    
    x = gcn(feature_matrix, adj_matrix )
    x = lbann.Relu(x,name="GCN1_activation") 

    x = gcn2(x, adj_matrix)
    x = lbann.Relu(x, name="GCN2_activation")
    
    x = gcn3 (x, adj_matrix)
    x = lbann.Relu(x, name="GCN3_activation")
   
    #----------------------------------
    # Apply Reduction on Node Features
    #----------------------------------
    
    #out_channel = node_features
    #print(node_features)
    average_vector = lbann.Constant(value = 1/num_vertices, num_neurons = str_list([1,num_vertices]), name="Average_Vector")
    x = lbann.MatMul(average_vector,x, name="Node_Feature_Reduction") # X is now a vector with output_channel dimensions 
    
    x = lbann.Reshape(x, dims= str_list([out_channel]), name="Squeeze")
    x = lbann.FullyConnected(x, num_neurons=64, name="hidden_layer_1")
    x = lbann.Relu(x, name="hidden_layer_1_activation")
    x = lbann.FullyConnected(x, num_neurons=num_classes, name="Output_Fully_Connected")
    
    #----------------------------------
    # Loss Function and Accuracy s
    #----------------------------------
    
    
    probs = lbann.Softmax(x, name="Softmax")
    loss = lbann.MeanSquaredError(x, target, name="Cross_Entropy_Loss")
    accuracy = lbann.CategoricalAccuracy(probs, target, name="Accuracy")

    layers = lbann.traverse_layer_graph(input_)
    
    print_model = lbann.CallbackPrintModelDescription() #Prints initial Model after Setup

    training_output = lbann.CallbackPrint( interval = 1,
                           print_global_stat_only = False) #Prints training progress
    gpu_usage = lbann.CallbackGPUMemoryUsage()
    
    timer = lbann.CallbackTimer()

    callbacks = [training_output, timer, print_model]

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
 
