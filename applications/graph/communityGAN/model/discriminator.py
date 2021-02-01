import math
import numpy as np
import lbann

from util import str_list

class Discriminator(lbann.modules.Module):

    def __init__(
            self,
            num_vertices,
            motif_size,
            embed_dim,
            learn_rate,
    ):
        super().__init__()
        self.num_vertices = num_vertices
        self.embed_dim = embed_dim
        self.learn_rate = learn_rate

        # Initialize weights
        # Note: The discriminator's probability estimate is
        #   D = 1 - exp(-sum_j(prod_i(d_ij)))
        # Treating the embeddings as i.i.d. random variables:
        #   D = 1 - exp( -embed_dim * d^motif_size )
        #   log(d) = log( -log(1-D) / embed_dim ) / motif_size
        # We initialize the embeddings in log-space so that the
        # discriminator's initial probability estimates have mean 0.5.
        mean = math.log( -math.log(1-0.5) / embed_dim ) / motif_size
        radius = math.log( -math.log(1-0.75) / embed_dim ) / motif_size - mean
        self.log_embedding_weights = lbann.Weights(
            initializer=lbann.UniformInitializer(
                min=mean-radius, max=mean+radius),
            name='discriminator_log_embeddings',
        )

        # Initialize cache for helper function
        self.triu_mask_cache = {}

    def get_log_embeddings(self, indices):
        return lbann.DistEmbedding(
            indices,
            weights=self.log_embedding_weights,
            num_embeddings=self.num_vertices,
            embedding_dim=self.embed_dim,
            sparse_sgd=True,
            learning_rate=self.learn_rate,
        )

    def forward(self, motif_size, motif_log_embeddings):
        """Predict whether a motif is real.

        @todo Numerically accurate computation of both log(D) and
        log(1-D).

        """

        # D = 1 - exp(-sum_j(prod_i(d_ij)))
        # log(1-D) = -sum_j(exp(sum_i(log(d_ij))))
        x = lbann.MatMul(
            lbann.Constant(value=1, num_neurons=str_list([1, motif_size])),
            motif_log_embeddings,
        )
        x = lbann.Exp(x)
        x = lbann.Reduction(x, mode='sum')
        x = lbann.Negative(x)
        log_not_prob = x

        # Convert log-probability to linear space
        # Note: D=-expm1(x) is accurate when D~0. When D~1, prefer
        # 1-D=exp(x).
        prob = lbann.Negative(lbann.Expm1(log_not_prob))

        return prob, log_not_prob

    def _triu(self, x, dims, k):
        if (dims, k) not in self.triu_mask_cache:
            vals = np.triu(np.full(dims, 1, dtype=int), k=k)
            w = lbann.Weights(
                initializer=lbann.ValueInitializer(values=str_list(np.nditer(vals))),
                optimizer=lbann.NoOptimizer(),
            )
            self.triu_mask_cache[(dims, k)] = lbann.WeightsLayer(
                dims=str_list(dims),
                weights=w,
            )
        return lbann.Multiply(x, self.triu_mask_cache[(dims, k)])
