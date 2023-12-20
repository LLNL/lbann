import lbann
from lbann.modules import Module, ChannelwiseFullyConnectedModule, ConvolutionModule
import lbann.modules


class GCN(Module):
    """
    Graph convolutional kernel
    """

    def __init__(
        self,
        num_nodes,
        num_edges,
        input_features,
        output_features,
        activation=lbann.Relu,
        distconv_enabled=True,
        num_groups=4,
    ):
        super().__init__()
        self._input_dims = input_features
        self._output_dims = output_features
        self._num_nodes = num_nodes
        self._num_edges = num_edges

    def forward(self, node_features, source_indices, target_indices):
        x = lbann.Gather(node_features, target_indices, axis=0)
        x = lbann.ChannelwiseFullyConnected(x, output_channel_dims=self._output_dims)
        x = self._activation(x)
        x = lbann.Scatter(x, source_indices, dims=self._ft_dims)
        return x


def create_model(num_nodes, num_edges, input_features, output_features, num_layers=3):
    """
    Create a GCN model
    """
    # Layer graph
    input_ = lbann.Input()
    split_indices = [0, num_nodes * input_features]
    split_indices += [split_indices[-1] + num_edges]
    split_indices += [split_indices[-1] + num_edges]
    split_indices += [split_indices[-1] + num_nodes]

    node_features = lbann.Reshape(
        lbann.Identity(input_), dims=[num_nodes, input_features]
    )

    source_indices = lbann.Reshape(lbann.Identity(input_), dims=[num_edges])
    target_indices = lbann.Reshape(lbann.Identity(input_), dims=[num_edges])
    label = lbann.Reshape(lbann.Identity(input_), dims=[num_nodes])

    x = GCN(
        num_nodes,
        num_edges,
        input_features,
        output_features,
        activation=lbann.Relu,
        distconv_enabled=False,
        num_groups=4,
    )(node_features, source_indices, target_indices)

    for _ in range(num_layers - 1):
        x = GCN(
            num_nodes,
            num_edges,
            input_features,
            output_features,
            activation=lbann.Relu,
            distconv_enabled=False,
            num_groups=4,
        )(x, source_indices, target_indices)

    # Loss function
    loss = lbann.CrossEntropy([x, label])

    # Metrics
    acc = lbann.CategoricalAccuracy([x, label])

    # Callbacks
    callbacks = [lbann.CallbackPrint(), lbann.CallbackTimer()]

    # Construct model
    return lbann.Model(
        num_epochs=1,
        layers=lbann.traverse_layer_graph(input_),
        objective_function=loss,
        metrics=[acc],
        callbacks=callbacks,
    )
