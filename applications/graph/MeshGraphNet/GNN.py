import lbann
from GNNComponents import MLP, GraphProcessor


def input_data_splitter(
    input_layer, num_nodes, num_edges, in_dim_node, in_dim_edge, out_dim
):
    """Takes a flattened sample from the Python DataReader and slices
    them according to the graph attributes.
    """

    split_indices = []
    start_index = 0
    node_feature_size = num_nodes * in_dim_node
    edge_feature_size = num_edges * in_dim_edge
    out_feature_size = num_nodes * out_dim

    split_indices.append(start_index)
    split_indices.append(split_indices[-1] + node_feature_size)
    split_indices.append(split_indices[-1] + edge_feature_size)
    split_indices.append(split_indices[-1] + num_edges)
    split_indices.append(split_indices[-1] + num_edges)
    split_indices.append(split_indices[-1] + out_feature_size)

    sliced_input = lbann.Slice(input_layer, axis=0, slice_points=split_indices)

    node_features = lbann.Reshape(
        lbann.Identity(sliced_input), dims=[num_nodes, in_dim_node]
    )
    edge_features = lbann.Reshape(
        lbann.Identity(sliced_input), dims=[num_edges, in_dim_edge]
    )
    source_node_indices = lbann.Reshape(lbann.Identity(sliced_input), dims=[num_edges])
    target_node_indices = lbann.Reshape(lbann.Identity(sliced_input), dims=[num_edges])

    out_features = lbann.Reshape(
        lbann.Identity(sliced_input), dims=[num_nodes, out_dim]
    )

    return (
        node_features,
        edge_features,
        source_node_indices,
        target_node_indices,
        out_features,
    )


def LBANN_GNN_Model(
    num_nodes,
    num_edges,
    in_dim_node,
    in_dim_edge,
    out_dim,
    out_dim_node=128,
    out_dim_edge=128,
    hidden_dim_node=128,
    hidden_dim_edge=128,
    hidden_layers_node=2,
    hidden_layers_edge=2,
    mp_iterations=15,
    hidden_dim_processor_node=128,
    hidden_dim_processor_edge=128,
    hidden_layers_processor_node=2,
    hidden_layers_processor_edge=2,
    norm_type=lbann.LayerNorm,
    hidden_dim_decoder=128,
    hidden_layers_decoder=2,
    num_epochs=10,
):
    # Set up model modules and associated weights
     
    node_encoder = MLP(
        in_dim=in_dim_node,
        out_dim=out_dim_node,
        hidden_dim=hidden_dim_node,
        hidden_layers=hidden_layers_node,
        norm_type=norm_type,
        name="graph_input_node_encoder",
    )

    edge_encoder = MLP(
        in_dim=in_dim_edge,
        out_dim=out_dim_edge,
        hidden_dim=hidden_dim_edge,
        hidden_layers=hidden_layers_edge,
        norm_type=norm_type,
        name="graph_input_edge_encoder",
    )

    # The graph processor currently only implements homogenous node graphs
    # so we do not distinguish between world and mesh nodes. LBANN supports 
    # heterogenous and multi-graphs in general

    # We also disable adaptive remeshing as that may require recomputing
    # the compute graph due to changing graph characteristics
    graph_processor = GraphProcessor(
        num_nodes=num_nodes,
        mp_iterations=mp_iterations,
        in_dim_node=out_dim_node,
        in_dim_edge=out_dim_edge,
        hidden_dim_node=hidden_dim_processor_node,
        hidden_dim_edge=hidden_dim_processor_edge,
        hidden_layers_node=hidden_layers_processor_node,
        hidden_layers_edge=hidden_layers_processor_edge,
        norm_type=norm_type,
    )

    node_decoder = MLP(
        in_dim=out_dim_node,
        out_dim=out_dim,
        hidden_dim=hidden_dim_decoder,
        hidden_layers=hidden_layers_decoder,
        norm_type=None,
        name="graph_input_node_decoder",
    )

    # Define LBANN Compute graph

    input_layer = lbann.Input(data_field="samples")

    (
        node_features,
        edge_features,
        source_node_indices,
        target_node_indices,
        out_features,
    ) = input_data_splitter(
        input_layer, num_nodes, num_edges, in_dim_node, in_dim_edge, out_dim
    )

    node_features = node_encoder(node_features)
    edge_features = edge_encoder(edge_features)

    node_features, _ = graph_processor(
        node_features, edge_features, source_node_indices, target_node_indices
    )

    calculated_features = node_decoder(node_features)

    loss = lbann.MeanSquaredError(calculated_features, out_features)

    # Define some of the usual callbacks

    training_output = lbann.CallbackPrint(interval=1, print_global_stat_only=False)
    gpu_usage = lbann.CallbackGPUMemoryUsage()
    timer = lbann.CallbackTimer()
    callbacks = [training_output, gpu_usage, timer]

    # Putting it all together and compile the model

    layers = lbann.traverse_layer_graph(input_layer)
    model = lbann.Model(
        num_epochs, layers=layers, objective_function=loss, callbacks=callbacks
    )
    return model
