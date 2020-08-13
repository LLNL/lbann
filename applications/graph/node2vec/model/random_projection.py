import lbann
import lbann.modules

import utils

def random_projection(indices, num_projections, projection_dim):

    # Expand input indices to get an index for each vector entry
    # Note: proj_indices(i) = index*projection_dim + i
    proj_indices = lbann.WeightedSum(
        indices,
        scaling_factors=utils.str_list(projection_dim),
    )
    iota = lbann.WeightsLayer(
        dims=utils.str_list(projection_dim),
        weights=lbann.Weights(
            initializer=lbann.ValueInitializer(
                values=utils.str_list(range(projection_dim))
            ),
            optimizer=lbann.NoOptimizer(),
        ),
    )
    proj_indices = lbann.Sum(
        lbann.Tessellate(
            lbann.Reshape(proj_indices, dims=utils.str_list([num_projections, 1])),
            dims=utils.str_list([num_projections, projection_dim]),
        ),
        lbann.Tessellate(
            lbann.Reshape(iota, dims=utils.str_list([1, projection_dim])),
            dims=utils.str_list([num_projections, projection_dim]),
        ),
    )

    # Apply hash function and convert to Gaussian distribution
    proj = lbann.UniformHash(proj_indices)
    ones = lbann.Constant(
        value=1,
        num_neurons=utils.str_list([num_projections, projection_dim]),
    )
    eps = 0.001
    proj = lbann.ErfInv(
        lbann.WeightedSum(
            proj,
            ones,
            scaling_factors=utils.str_list([2*(1-eps), -(1-eps)]),
        )
    )
    proj = lbann.InstanceNorm(proj)
    proj = lbann.WeightedSum(
        proj,
        scaling_factors=utils.str_list(1/projection_dim),
    )
    return proj

class ChannelwiseFullyConnectedAutoencoder(lbann.modules.Module):

    def __init__(self, input_dim, output_dim, hidden_dims=[]):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = utils.make_iterable(hidden_dims)

        self.weights = [
            lbann.Weights(initializer=lbann.HeNormalInitializer())
            for _ in range(len(self.hidden_dims))
        ]
        self.weights.append(
            lbann.Weights(initializer=lbann.HeNormalInitializer())
        )

    def encode(self, x):
        x = lbann.Reshape(x, dims=utils.str_list([-1, self.input_dim]))
        for i, dim in enumerate(self.hidden_dims):
            x = lbann.ChannelwiseFullyConnected(
                x,
                weights=self.weights[i],
                output_channel_dims=dim,
                bias=False,
            )
            x = lbann.Relu(x)
        x = lbann.ChannelwiseFullyConnected(
            x,
            weights=self.weights[-1],
            output_channel_dims=self.output_dim,
            bias=False,
        )
        return x

    def decode(self, x):
        x = lbann.Reshape(x, dims=utils.str_list([-1, self.output_dim]))
        for i in range(len(self.hidden_dims)-1, 0, -1):
            x = lbann.ChannelwiseFullyConnected(
                x,
                weights=self.weights[i+1],
                output_channel_dims=self.hidden_dims[i],
                transpose=True,
                bias=False,
            )
            x = lbann.Relu(x)
        x = lbann.ChannelwiseFullyConnected(
            x,
            weights=self.weights[0],
            output_channel_dims=self.input_dim,
            transpose=True,
            bias=False,
        )
        return x

    def forward(self, x):
        return self.decode(self.encode(x))
