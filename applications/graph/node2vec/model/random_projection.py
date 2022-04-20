import numpy as np
import lbann
import lbann.modules

import utils

def random_projection(indices, num_projections, projection_dim):

    # Expand input indices to get an index for each vector entry
    # Note: proj_indices(i) = index*projection_dim + i
    proj_indices = lbann.WeightedSum(
        indices,
        scaling_factors=projection_dim,
    )
    iota = lbann.WeightsLayer(
        dims=.projection_dim,
        weights=lbann.Weights(
            initializer=lbann.ValueInitializer(
                values=range(projection_dim)
            ),
            optimizer=lbann.NoOptimizer(),
        ),
    )
    proj_indices = lbann.Sum(
        lbann.Tessellate(
            lbann.Reshape(proj_indices, dims=[num_projections, 1]),
            dims=[num_projections, projection_dim],
        ),
        lbann.Tessellate(
            lbann.Reshape(iota, dims=[1, projection_dim]),
            dims=[num_projections, projection_dim],
        ),
    )

    # Apply hash function and convert to Gaussian distribution
    proj = lbann.UniformHash(proj_indices)
    ones = lbann.Constant(
        value=1,
        num_neurons=[num_projections, projection_dim],
    )
    eps = 0.001
    proj = lbann.ErfInv(
        lbann.WeightedSum(
            proj,
            ones,
            scaling_factors=[2*(1-eps), -(1-eps)],
        )
    )
    proj = lbann.InstanceNorm(proj)
    proj = lbann.WeightedSum(
        proj,
        scaling_factors=1/projection_dim,
    )
    return proj

def mean_squared_error(
        data_dim,
        sequence_length,
        source_sequence,
        target_sequence,
        scale_decay=0.8,
):

    # Compute inner product between source and target vectors
    # Note: Inner products are computed for each (x,y) pair and a
    # weighted sum is computed. The scaling factors sum to 1 and decay
    # exponentially as x and y get further apart in the sequence.
    prods = lbann.MatMul(
        source_sequence,
        target_sequence,
        transpose_b=True,
    )
    scale_dims = (sequence_length,sequence_length)
    scales = np.zeros(scale_dims)
    for i in range(sequence_length):
        for j in range(sequence_length):
            if i != j:
                scales[i,j] = (
                    (1-scale_decay)/(2*scale_decay)
                    * scale_decay**np.abs(j-i)
                )
    scales = lbann.Weights(
        initializer=lbann.ValueInitializer(values=np.nditer(scales)),
        optimizer=lbann.NoOptimizer(),
    )
    scales = lbann.WeightsLayer(dims=scale_dims, weights=scales)
    prods = lbann.MatMul(
        lbann.Reshape(prods, dims='1 -1'),
        lbann.Reshape(scales, dims='1 -1'),
        transpose_b=True,
    )
    prods = lbann.Reshape(prods, dims='1')

    # MSE(x,y) = ( norm(x)^2 + norm(y)^T - 2*prod(x,y) ) / dim(x)
    scale = 1 / (data_dim * sequence_length)
    return lbann.WeightedSum(
        lbann.L2Norm2(source_sequence),
        lbann.L2Norm2(target_sequence),
        prods,
        scaling_factors=[scale, scale, -2*scale]
    )

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
        x = lbann.Reshape(x, dims=[-1, self.input_dim])
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
        x = lbann.Reshape(x, dims=[-1, self.output_dim])
        for i in range(len(self.hidden_dims)):
            x = lbann.ChannelwiseFullyConnected(
                x,
                weights=self.weights[-i-1],
                output_channel_dims=self.hidden_dims[-i-1],
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
