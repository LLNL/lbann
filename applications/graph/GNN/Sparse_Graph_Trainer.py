import lbann 
from lbann.util import str_list 
from lbann.modules.graph import GINConv, GCNConv, GraphConv, GatedGraphConv
from lbann.modules.graph import GraphVertexData
from graph_data_util import lbann_Graph_Data

def GINConvLayer(X,A):
    """An example GIN kernel with 4 layer deep sequential nn.  
    Args:
        X (GraphVertexData): Contains all the node feaures of the graph 
        A (Layer): Adjancency matrix layer. Should have the shape: 
                   (num_nodes, num_nodes)
    Returns: 
        (GraphVertexData): Returns the new embedding of the node features 
    """
    FC = lbann.modules.FullyConnectedModule
    sequential_nn = \
                    [FC(128), 
                     lbann.Relu, 
                     FC(64),
                     lbann.Relu,
                     FC(32),
                     lbann.Relu,
                     FC(16),
                     lbann.Relu]
    out_channel = 16

    gin = GINConv(sequential_nn, output_channels = out_channel)
    return gin(X,A)


def GCNConvLayer(X,A):
    """An example 2-layer GCN kernel.
    Args:
        X (GraphVertexData): Contains all the node feaures of the graph
        A (Layer): Adjancency matrix layer. Should have the shape: 
                   (num_nodes, num_nodes)
    Returns: 
        (GraphVertexData): Returns the new embedding of the node features
    """
    input_channels_1 = X.shape[1]
    out_channels_1 = 8
    input_channels_2 = out_channels_1
    out_channels_2 = 16

    gcn_1 = GCNConv(input_channels_1,out_channels_1,
                    bias = True,
                    activation = lbann.Relu,
                    name = 'GCN_1',
                    data_layout = 'data_parallel')
    gcn_2 = GCNConv(input_channels_2,out_channels_2,
                    bias = True, 
                    activation = lbann.Relu,
                    name = 'GCN_2',
                    data_layout = 'data_parallel')
    X = gcn_1(X,A)
    return  gcn_2(X,A)

   
def GraphConvLayer(X,A):
    """An example 2-layer Graph kernel.
    Args:
        X (GraphVertexData): Contains all the node feaures of the graph
        A (Layer): Adjancency matrix layer. Should have the shape: 
                   (num_nodes, num_nodes)
    Returns: 
        (GraphVertexData): Returns the new embedding of the node features
    """
    input_channels_1 = X.shape[1]
    out_channels_1 = 8 
    input_channels_2 = out_channels_1
    out_channels_2 = 16
    
    graph_1 = GraphConv(input_channels_1, out_channels_1,
                        bias = True,
                        activation = lbann.Relu,
                        name = 'Graph_kernel_1',
                        data_layout = 'data_parallel')
    graph_2 = GraphConv(input_channels_2, out_channels_2,
                        bias = True,
                        activation = lbann.Relu, 
                        name = 'Graph_Kernel_2',
                        data_layout = 'data_parallel')

    X = graph_1(X,A)
    return graph_2(X,A)

def GATConvLayer(X,A):
    """An example single layer GatedGraph kernel.
    Args:
        X (GraphVertexData): Contains all the node feaures of the graph
        A (Layer): Adjancency matrix layer. Should have the shape: 
                   (num_nodes, num_nodes)
    Returns: 
        (GraphVertexData): Returns the new embedding of the node features
    """
    
    output_channels = 8
    num_layers = 3
    name = 'GatedGraph'
    data_layout = 'data_parallel' 

    graph_kernel = GatedGraphConv(output_channels,
                                  num_layers = num_layers,
                                  name = name, 
                                  data_layout = data_layout)
    return graph_kernel(X,A)

def make_model(num_vertices = None, 
               node_features = None, 
               num_classes = None,
               dataset = None,
               kernel_type = 'GCN',
               callbacks = None,
               num_epochs = 1):
    '''Construct a model DAG using one of the Graph Kernels

    Args:
        num_vertices (int): Number of vertices of each graph (default: None) 
        node_features (int): Number of features per noded (default: None)
        num_classes (int): Number of classes as targets (default: None)
        dataset (str): Preset data set to use. Either a datset parameter has to be 
                       supplied or all of num_vertices, node_features, and 
                       num_classes have to be supplied. (default: None) 
        kernel_type (str): Graph Kernel to use in model. Expected one of 
                            GCN, GIN, Graph, or GatedGraph (deafult: GCN)
        callbacks (list): Callbacks for the model. If set to None the model description, 
                          GPU usage, training_output, and timer is reported. 
                          (default: None)                    
        num_epochs (int): Number of epochs to run (default: 1)
    Returns:
        (lbann Model Object: A model object with the supplied callbacks, dataset
                               presets, and graph kernels. 
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
    # Input should have atleast two children since the target is classification 
    
    data = lbann_Graph_Data(input_,num_vertices, node_features,num_classes)
    
    feature_matrix = data.x 
    adj_matrix = data.adj 
    target = data.y 
   
    #----------------------------------
    # Perform Graph Convolution
    #----------------------------------

    if kernel_type == 'GIN':
        x = GINConvLayer(feature_matrix, adj_matrix) 
    elif kernel_type == 'GCN':
        x = GCNConvLayer(feature_matrix, adj_matrix)
    elif kernel_type == 'Graph':
        x = GraphConvLayer(feature_matrix, adj_matrix) 
    elif kernel_type == 'GatedGraph':
        x = GATConvLayer(feature_matrix, adj_matrix) 
    else:
        ValueError('Invalid Graph kernel specifier "{}" recieved. Expected one of:\
                    GIN,GCN,Graph or GatedGraph'.format(kernel_type))
    
    out_channel = x.shape[1]
    #----------------------------------
    # Apply Reduction on Node Features
    #----------------------------------

    average_vector = lbann.Constant(value = 1/num_vertices, 
                                    num_neurons = str_list([1,num_vertices]),
                                    name="Average_Vector")
    x = x.get_mat(out_channel)
    
    x = lbann.MatMul(average_vector,x, name="Node_Feature_Reduction") 
    
    # X is now a vector with output_channel dimensions 
    
    x = lbann.Reshape(x, dims = str_list([out_channel]), name = "Squeeze")
    x = lbann.FullyConnected(x, num_neurons = 64, name = "hidden_layer_1")
    x = lbann.Relu(x, name = "hidden_layer_1_activation")
    x = lbann.FullyConnected(x, num_neurons = num_classes,
                                name="Output_Fully_Connected")
    
    #----------------------------------
    # Loss Function and Accuracy s
    #----------------------------------
    
    
    probs = lbann.Softmax(x, name="Softmax")
    loss = lbann.CrossEntropy(probs, target, name="Cross_Entropy_Loss")
    accuracy = lbann.CategoricalAccuracy(probs, target, name="Accuracy")

    layers = lbann.traverse_layer_graph(input_)
    
    if callbacks is None:
        print_model = lbann.CallbackPrintModelDescription() #Prints initial Model after Setup
        training_output = lbann.CallbackPrint( interval = 1,
                           print_global_stat_only = False) #Prints training progress
        gpu_usage = lbann.CallbackGPUMemoryUsage()
        timer = lbann.CallbackTimer()
        callbacks = [print_model, training_output, gpu_usage, timer]
    else:
        if isinstance (callbacks, list):
            callbacks = callbacks

    metrics = [lbann.Metric(accuracy, name='accuracy', unit="%")]

    model = lbann.Model(num_epochs, 
                       layers = layers,
                       objective_function = loss,
                       metrics = metrics, 
                       callbacks = callbacks
                       )
    return model

if __name__ == '__main__':
    # Quick check to see if model generates correctly
    model_1 = make_model(dataset="MNIST", kernel_type = 'GIN')
    model_1 = make_model(dataset="MNIST", kernel_type = 'GCN')
    model_1 = make_model(dataset="MNIST", kernel_type = 'Graph')
    model_1 = make_model(dataset="MNIST", kernel_type = 'GatedGraph')
