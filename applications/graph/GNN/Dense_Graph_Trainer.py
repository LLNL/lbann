import lbann
from lbann.util import str_list
from lbann.modules.graph import DenseGCNConv, DenseGraphConv


def DGCN_layer(feature_matrix,adj_matrix, node_features):
    """An example 3 layer GCN kernel.
    Args:
        feature_matrix (Layer): Node feature layer. Should have the shape:
                                (num_nodes, node_features)
        adj_matrix (Layer): Adjancency matrix layer. Should have the shape:
                            (num_nodes, num_nodes)
        node_features (int): The number of features per node
    Returns:
        (Layer): Returns the new embedding of the node features
    """
    out_channel_1 = 1024
    out_channel_2 = 512
    out_channel_3 = 256

    gcn1 = DenseGCNConv(input_channels = node_features, output_channels = out_channel_1)
    gcn2 = DenseGCNConv(input_channels = out_channel_1, output_channels = out_channel_2)
    gcn3 = DenseGCNConv(input_channels = out_channel_2, output_channels = out_channel_3)

    out_channel = out_channel_3

    x = gcn1(feature_matrix, adj_matrix )
    x = lbann.Relu(x,name="DGCN1_activation")

    x = gcn2(x, adj_matrix)
    x = lbann.Relu(x, name="DGCN2_activation")

    x = gcn3 (x, adj_matrix)
    x = lbann.Relu(x, name="DGCN3_activation")
    return x


def DGraph_Layer(feature_matrix,adj_matrix, node_features):
    """An example 3 layer Graph kernel.
    Args:
        feature_matrix (Layer): Node feature layer. Should have the shape:
                                (num_nodes, node_features)
        adj_matrix (Layer): Adjancency matrix layer. Should have the shape:
                            (num_nodes, num_nodes)
        node_features (int): The number of features per node
    Returns:
        (Layer): Returns the new embedding of the node features
    """
    out_channel_1 = 1024
    out_channel_2 = 512
    out_channel_3 = 256

    gcn1 = DenseGraphConv(input_channels = node_features, output_channels = out_channel_1)
    gcn2 = DenseGraphConv(input_channels = out_channel_1, output_channels = out_channel_2)
    gcn3 = DenseGraphConv(input_channels = out_channel_2, output_channels = out_channel_3)

    out_channel = out_channel_3

    x = gcn1(feature_matrix, adj_matrix )
    x = lbann.Relu(x,name="DGraph1_activation")

    x = gcn2(x, adj_matrix)
    x = lbann.Relu(x, name="DGraph2_activation")

    x = gcn3 (x, adj_matrix)
    x = lbann.Relu(x, name="DGraph3_activation")
    return x


def make_model(num_vertices = None,
               node_features = None,
               num_classes = None,
               kernel_type = 'GCN',
               callbacks = None,
               num_epochs = 1):
    '''Construct a model DAG using one of the Graph Kernels

    Args:
        num_vertices (int): Number of vertices of each graph (default: None)
        node_features (int): Number of features per noded (default: None)
        num_classes (int): Number of classes as targets (default: None)
        kernel_type (str): Graph Kernel to use in model. Expected one of
                            GCN, or Graph (deafult: GCN)
        callbacks (list): Callbacks for the model. If set to None the model description,
                          GPU usage, training_output, and timer is reported.
                          (default: None)
        num_epochs (int): Number of epochs to run (default: 1)
    Returns:
        (lbann Model Object: A model object with the supplied callbacks, dataset
                               presets, and graph kernels.
    '''

    num_vertices = 100
    num_classes = 2
    node_features = 3

    assert num_vertices is not None
    assert num_classes is not None
    assert node_features is not None


    #----------------------------------
    # Reshape and Slice Input Tensor
    #----------------------------------

    input_ = lbann.Input(data_field='samples')

    # Input dimensions should be (num_vertices * node_features + num_vertices^2 + num_classes )
    # input should have atleast two children since the target is classification

    sample_dims = num_vertices*node_features + (num_vertices ** 2) + num_classes
    graph_dims = num_vertices*node_features + (num_vertices ** 2)
    feature_matrix_size = num_vertices * node_features

    graph_input = lbann.Slice(input_, axis = 0 ,
                              slice_points = str_list([0,feature_matrix_size,graph_dims, sample_dims]),
                              name = "Graph_Input")


    feature_matrix = lbann.Reshape(graph_input,
                                   dims = str_list([num_vertices, node_features]),
                                   name="Node_features")

    adj_matrix = lbann.Reshape(graph_input,
                               dims = str_list([num_vertices,num_vertices]),
                               name="Adj_Mat")

    target = lbann.Identity(graph_input, name="Target")
    target = lbann.Reshape(target, dims=str(num_classes))

    #----------------------------------
    # Perform Graph Convolution
    #----------------------------------

    if kernel_type == 'GCN':
        x = DGCN_layer(feature_matrix, adj_matrix, node_features)
    elif kernel_type == 'Graph':
        x = DGraph_Layer(feature_matrix, adj_matrix, node_features)
    else:
        ValueError('Invalid Graph kernel specifier "{}" recieved. Expected one of:\
                    GCN or Graph'.format(kernel_type))
    out_channel = 256
    #----------------------------------
    # Apply Reduction on Node Features
    #----------------------------------

    average_vector = lbann.Constant(value = 1/num_vertices, num_neurons = str_list([1,num_vertices]), name="Average_Vector")
    x = lbann.MatMul(average_vector,x, name="Node_Feature_Reduction") # X is now a vector with output_channel dimensions

    x = lbann.Reshape(x, dims= str_list([out_channel]), name="Squeeze")
    x = lbann.FullyConnected(x, num_neurons=256, name="hidden_layer_1")
    x = lbann.Relu(x, name="hidden_layer_1_activation")
    x = lbann.FullyConnected(x, num_neurons=num_classes, name="Output_Fully_Connected")

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
    model = make_model(dataset="MNIST")
    model = make_model(dataset="MNIST", kernel_type = 'Graph')
