import lbann
from lbann.modules import Module, ChannelwiseFullyConnectedModule, ConvolutionModule
import lbann.modules
import math


def ContractHeads(lbann_graph_layer, shape):
    """
    A utility function that contracts the rows of a (N, M, H) matrix to an (N, M) matrix using grouped 2D convolution.
    The contration computes the average along the first dimension so the output is scaled by 1 / H.

    Args:
        lbann_graph_layer (layer): Graph layer tensor with shape (N, M, H)

        shape ((int, int, int)): Shape of graph layer tensor

    Returns:
        (Layer): Contracted and rescaled output with shape (N, M)
    """
    num_nodes, output_channels, num_heads = shape
    weights = lbann.Weights(
        initializer=lbann.ConstantInitializer(value=1 / num_heads),
        optimizer=lbann.NoOptimizer(),
    )
    kernel_shape = (1, num_heads)
    contraction = lbann.Convolution(
        num_dims=2,
        output_channels=num_nodes,
        kernel_size=kernel_shape,
        stride=1,
        padding=0,
        groups=num_nodes,
        has_bias=False,
        weights=weights,
    )
    output = lbann.Reshape(contraction, dims=[num_nodes, output_channels])
    return output


class GAT(Module):
    """Graph Attention Network layer. For kernel details, see:

    https://arxiv.org/abs/1710.10903

    """

    global_count = 0

    def __init__(
        self,
        input_channels,
        output_channels,
        num_nodes,
        num_edges,
        num_heads=1,
        name=None,
    ):
        """Initialize GatedGraph layer
        Args:
            input_channels (int): The size of the input node features
            output_channels (int): The output size of the node features
            num_nodes (int): Number of vertices in the graph
            num_edges (int): Number of edges in the graph
            num_heads (int): Number of attention heads (default: 1)
            name (str): Name of the layers and prefix to use for the layers.
            data_layout (str): Data layout (default: data parallel)
        """
        super().__init__()

        # Add Name for the components for the layer
        GAT.global_count += 1
        self.name = name if name else "GAT_{}".format(GAT.global_count)
        # Add variables
        self.output_channel_size = output_channels
        self.input_channel_size = input_channels
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.num_heads = num_heads

        weights = lbann.Weights(
            initializer=lbann.UniformInitializer(
                min=-1 / (math.sqrt(output_channels)),
                max=1 / (math.sqrt(output_channels)),
            )
        )
        self.W_k = ChannelwiseFullyConnectedModule(
            self.output_channel_size * num_heads,
            bias=False,
            weights=[weights],
            name=f"{self.name}_nn_{1}",
        )

        self.a_vec = ConvolutionModule(
            num_dims=1,
            out_channels=self.num_nodes,
            kernel_size=[2 * self.output_channel_size, 1],
            groups=self.num_nodes,
            bias=False,
            name=f"{self.name}_nn_{2}",
        )

    def forward(
        self, node_feature_mat, source_indices, target_indices, reduction="concat"
    ):
        """Call GATGraphConv
        Args:
            node_feature_mat (Layer): Node feature matrix with the shape of (num_nodes, input_channels)
            source_indices (Layer): Source node indices of the edges with shape (num_edges)
            target_indices (Layer): Target node indices of the edges with shape (num_edges)
            reduction (string: [concat| average]): The type of reductions to use for multiple heads
        Returns:
            (Layer) : The output after kernel ops. The shape of the layer is
                (num_nodes, num_heads * num_output_channels) if reduction is "concat"
                (num_nodes, num_output_channels) if reduction is "average"
        """
        # (N x [self.output_channel * self.num_heads])
        transform_node_features = self.W_nn(
            node_feature_mat, name=f"{self.name}_transform"
        )
        # (E x [self.output_channel * self.num_heads])
        e_i = lbann.Gather(transform_node_features, source_indices, axis=0)
        e_j = lbann.Gather(transform_node_features, target_indices, axis=0)
        # (E x self.output_channel x self.num_heads)
        e_i = lbann.Reshape(
            e_i, dims=[self.num_edges, self.output_channel_size, self.num_heads]
        )
        e_j = lbann.Reshape(
            e_j, dims=[self.num_edges, self.output_channel_size, self.num_heads]
        )
        # (E x 2 * self.output_channel x self.num_heads)
        messages = lbann.Concatenation([e_i, e_j], axis=1)
        # (E x self.num_heads)
        m_ij = lbann.Reshape(
            self.a_vec(messages), dims=[self.num_edges, self.num_heads]
        )
        m_ij = lbann.ExpOperator(lbann.LeakyRelu(m_ij, negative_slope=0.02))
        # (N x self.num_heads)
        contraction = lbann.Scatter(m_ij, target_indices, axis=0)
        # (N x 1 x self.num_heads)
        broadcast = lbann.Reshape(contraction, dims=[self.num_nodes, 1, self.num_heads])
        # (E x 1 x self.num_heads)
        broadcast = lbann.Gather(broadcast, target_indices, axis=1)
        # (E x self.output_channel_size x self.num_heads)
        broadcast = lbann.Tessellate(
            broadcast, dims=[self.num_edges, self.output_channel_size, self.num_heads]
        )
        # (E x self.output_channel_size x self.num_heads)
        normalize = lbann.Scatter(broadcast, source_indices, axis=0)
        alpha_ij = lbann.DivideOperator(m_ij, normalize)

        h_ij = lbann.MultiplyOperator(alpha_ij, e_j)

        h_i = lbann.Scatter(h_ij, source_indices)

        if reduction.lower() == "concat":
            node_feature_mat = lbann.Reshape(h_i)
        elif reduction.lower() == "average":
            node_feature_mat = ContractHeads(
                h_i, (self.num_nodes, self.output_channel_size, self.num_heads)
            )
        else:
            raise ValueError("Expected reduction arguments are: concat or average")

        return node_feature_mat
