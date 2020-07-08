import lbann
from Graph_Kernels import GCN_Layer
from lbann.util import str_list

class lbann_Data_Mat:
    def __init__(self, list_of_layers, layer_size):
        self.size = (len(list_of_lbann_layers), layer_size)
        self.layers = list_of_layers
    
    def __getitem__(self, row):
        return self.layers[row]

class lbann_Graph_Data:
    def __init__(self, input_layer, num_vertices, num_features, num_classes):
        self.num_vertices = num_vertices 
        self.num_features = num_features 
        self.num_classes = num_classes 

        self.x, self.adj, self.y = gen_data()

    def generate_slice_points (self):
        slices_points = [for i in range(self.num_vertices * self.num_features + 1, self.num_features)]
        adj_mat = slice_points[-1] + self.num_vertices * self.num_vertices 
        slices_points.append(adj_mat)
        targets = slice_points[-1] + self.num_classes 
        slice_points.append(targets)
        return str_list(slice_points)
    
    def gen_data(self, input_layer):
        slice_points = generate_slice_points()
        sliced_graph  = lbann.Slice(input_layer, axis = 0, slice_points = slice_points, name="Sliced_Input")

        node_features = [] 

        for i in range(self.num_vertices):
            temp = lbann.Identity(sliced_graph)
            node_features.append(lbann.Reshape(temp, dims=str(self.num_vertices)))

        x = lbann_Data_Mat(node_features, self.num_features)
        
        adj_mat_in = lbann.Identity(sliced_graph) 
        adj_mat = lbann.Reshape(adj_mat_in, dims = str_list([self.num_vertices, self.num_vertices])) 

        y_ind = lbann.Identity(sliced_graph)
        y = lbann.Reshape(sliced_graph, dims=str(self.num_classes))
        return x, adj_mat, y 




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

        elif dataset == 'Synthetic':
            num_vertices = 3
            num_classes = 2
            node_features = 2
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
    target = lbann.Reshape(target, dims="10")
    #----------------------------------
    # Perform Graph Convolution
    #----------------------------------

    # To Do: Implement lbann.GCN()
    # 
    # x = lbann.GCN(feature_matrix, adj_matrix, output_channels = N) # X is now the feature_matrix of shape num_vertices x output_channels 
    #
    
   # print("Warning: Not using GCN layer and forwaring orignal feature matrix to reduction step. Should only use this when testing dataset / data reader")
    
    out_channel_1 = 1024
    out_channel_2 = 256
    out_channel_3 = 128
    
    gcn = GCN_Layer(input_channels = node_features, output_channels = out_channel_1)
    gcn2 = GCN_Layer(input_channels = out_channel_1, output_channels = out_channel_2)
    gcn3 = GCN_Layer(input_channels = out_channel_2, output_channels = out_channel_3)
    
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
 
