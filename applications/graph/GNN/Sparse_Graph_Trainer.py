import lbann
from lbann.modules.graph import GINConv, GCNConv, GraphConv, GatedGraphConv
from itertools import accumulate


def Graph_Data_Parser(_lbann_input_,
                      num_nodes,
                      node_feature_size,
                      max_edges,
                      num_classes = 1):
    """ A parser for graph structured data with node
        features, source and target node indices (COO)
        format, and a target

    Args:
        _lbann_input_ (Layer): The input layer of the LBANN model
        num_nodes (int): The maximum number of nodes in the dataset
        node_features_size (int): The dimensionality of the node features matrix
        max_edges (int): The maximum number of edges in the dataset
        num_classes (int): The number of classes in the target or 1 for
                           regression (default : 1)
    Returns:
        (dictionary) Returns a dictionary with the keys: node_features, source_indices,
                     target_indices, and targets
    """
    slice_points = [0, num_nodes*node_feature_size, max_edges, max_edges, num_classes]
    shifted_slice_points = list(accumulate(slice_points))
    sliced_input = lbann.Slice(_lbann_input_,
                               slice_points=shifted_slice_points,
                               name="Sliced_Graph_Input")
    node_features = lbann.Reshape(lbann.Identity(sliced_input),
                                  dims=[num_nodes, node_feature_size],
                                  name="Node_Feature_Matrix")
    source_indices = lbann.Identity(sliced_input)
    target_indices = lbann.Identity(sliced_input)
    targets = lbann.Identity(sliced_input)

    graph_data = {"node_features":node_features,
                  "source_indices":source_indices,
                  "target_indices":target_indices,
                  "target":targets}
    return graph_data



def GINConvLayer(node_features,
                 source_indices,
                 target_indices,
                 num_nodes,
                 num_edges,
                 input_channels,
                 output_channels):
    """An example GIN kernel with 4 layer deep sequential nn.
    Args:
        node_feature (Layer): Node feature matrix with the shape of (num_nodes,input_channels)
        source_indices (Layer): Source node indices of the edges with shape (num_nodes)
        target_indices (Layer): Target node indices of the edges with shape (num_nodes)
        num_nodes (int): Number of vertices in the graph
        input_channels (int): The size of the input node features
        output_channels (int): The number of output channels of the node features
    Returns:
        (GraphVertexData): Returns the new embedding of the node features
    """
    FC = lbann.modules.ChannelwiseFullyConnectedModule
    sequential_nn = \
                    [FC(128),
                     lbann.Relu,
                     FC(64),
                     lbann.Relu,
                     FC(32),
                     lbann.Relu,
                     FC(output_channels),
                     lbann.Relu]

    gin = GINConv(sequential_nn,
                  input_channels = input_channels,
                  output_channels = output_channels,
                  num_nodes = num_nodes,
                  num_edges = num_edges)
    return gin(node_features, source_indices, target_indices)


def GCNConvLayer(node_features,
                 source_indices,
                 target_indices,
                 num_nodes,
                 num_edges,
                 input_channels,
                 output_channels):
    """An example 2-layer GCN kernel.
    Args:
        node_feature (Layer): Node feature matrix with the shape of (num_nodes,input_channels)
        source_indices (Layer): Source node indices of the edges with shape (num_nodes)
        target_indices (Layer): Target node indices of the edges with shape (num_nodes)
        num_nodes (int): Number of vertices in the graph
        input_channels (int): The size of the input node features
        output_channels (int): The number of output channels of the node features
    Returns:
        (Layer) : The resultant node features after message passing kernel ops
    """
    input_channels_1 = input_channels
    out_channels_1 = 8
    input_channels_2 = out_channels_1
    out_channels_2 = output_channels

    gcn_1 = GCNConv(input_channels_1,out_channels_1,
                    num_nodes,
                    bias = True,
                    activation = lbann.Relu,
                    name = 'GCN_1')
    gcn_2 = GCNConv(input_channels_2,out_channels_2,
                    num_nodes,
                    bias = True,
                    activation = lbann.Relu,
                    name = 'GCN_2')
    X = gcn_1(node_features,source_indices, target_indices)
    return  gcn_2(X,source_indices, target_indices)


def GraphConvLayer(node_features,
                   source_indices,
                   target_indices,
                   num_nodes,
                   num_edges,
                   input_channels,
                   output_channels):
    """An example 2-layer Graph kernel.
    Args:
        node_feature (Layer): Node feature matrix with the shape of (num_nodes,input_channels)
        source_indices (Layer): Source node indices of the edges with shape (num_nodes)
        target_indices (Layer): Target node indices of the edges with shape (num_nodes)
        num_nodes (int): Number of vertices in the graph
        input_channels (int): The size of the input node features
        output_channels (int): The number of output channels of the node features
    Returns:
        (Layer) : The resultant node features after message passing kernel ops
    """
    input_channels_1 = input_channels
    out_channels_1 = 8
    input_channels_2 = out_channels_1
    out_channels_2 = output_channels

    graph_1 = GraphConv(input_channels_1, out_channels_1,
                        num_nodes,
                        bias = True,
                        activation = lbann.Relu,
                        name = 'Graph_kernel_1')
    graph_2 = GraphConv(input_channels_2, out_channels_2,
                        num_nodes,
                        bias = True,
                        activation = lbann.Relu,
                        name = 'Graph_Kernel_2')

    X = graph_1(node_features,source_indices, target_indices)
    return graph_2(X,source_indices, target_indices)

def GATConvLayer(node_features,
                 source_indices,
                 target_indices,
                 num_nodes,
                 num_edges,
                 input_channels,
                 output_channels):
    """An example single layer GatedGraph kernel.
    Args:
        node_feature (Layer): Node feature matrix with the shape of (num_nodes,input_channels)
        source_indices (Layer): Source node indices of the edges with shape (num_nodes)
        target_indices (Layer): Target node indices of the edges with shape (num_nodes)
        num_nodes (int): Number of vertices in the graph
        input_channels (int): The size of the input node features
        output_channels (int): The number of output channels of the node features
    Returns:
        (Layer) : The resultant node features after message passing kernel ops
    """
    num_layers = 3
    name = 'GatedGraph'
    data_layout = 'data_parallel'

    graph_kernel = GatedGraphConv(input_channels, output_channels,
                                  num_nodes,
                                  num_layers = num_layers,
                                  name = name)
    return graph_kernel(node_features,source_indices, target_indices)

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
                            GCN, GIN, Graph, or GatedGraph (deafult: GCN)
        callbacks (list): Callbacks for the model. If set to None the model description,
                          GPU usage, training_output, and timer is reported.
                          (default: None)
        num_epochs (int): Number of epochs to run (default: 1)
    Returns:
        (lbann.Model) : A model object with the supplied callbacks, dataset
                               presets, and graph kernels.
    '''

    num_vertices = 100
    num_classes = 2
    node_feature_size = 3
    max_edges = 415

    #----------------------------------
    # Reshape and Slice Input Tensor
    #----------------------------------

    input_ = lbann.Input(data_field='samples')

    # Input dimensions should be (num_vertices * node_features + num_vertices^2 + num_classes )

    data = Graph_Data_Parser(input_,
                             num_vertices,
                             node_feature_size,
                             max_edges,
                             num_classes)

    feature_matrix = data['node_features']
    source_indices = data['source_indices']
    target_indices = data['target_indices']
    target = data['target']

    #----------------------------------
    # Select Graph Convolution
    #----------------------------------



    output_channels = 16
    graph_kernel_op = None
    if kernel_type == 'GIN':
        graph_kernel_op = GINConvLayer
    elif kernel_type == 'GCN':
        graph_kernel_op = GCNConvLayer
    elif kernel_type == 'Graph':
        graph_kernel_op = GraphConvLayer
    elif kernel_type == 'GatedGraph':
        graph_kernel_op = GATConvLayer
    else:
        raise ValueError('Invalid Graph kernel specifier "{}" recieved. Expected one of:\
                    GIN,GCN,Graph or GatedGraph'.format(kernel_type))
    #----------------------------------
    # Perform Graph Convolution
    #----------------------------------

    x = graph_kernel_op(feature_matrix,
                        source_indices,
                        target_indices,
                        num_vertices,
                        max_edges,
                        node_feature_size,
                        output_channels)
    #----------------------------------
    # Apply Reduction on Node Features
    #----------------------------------

    average_vector = lbann.Constant(value = 1/num_vertices,
                                    num_neurons = [1,num_vertices],
                                    name="Average_Vector")

    x = lbann.MatMul(average_vector,x, name="Node_Feature_Reduction")

    # X is now a vector with output_channel dimensions

    x = lbann.Reshape(x, dims = [output_channels], name = "Squeeze")
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
