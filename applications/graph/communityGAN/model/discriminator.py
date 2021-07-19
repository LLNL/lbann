import math
import lbann
import numpy as np

from util import str_list

class Discriminator(lbann.modules.Module):

    def __init__(
            self,
            num_vertices,
            motif_size,
            embed_dim,
            learn_rate,
            embeddings_device='CPU',
            initial_embeddings=None,
    ):
        super().__init__()
        self.num_vertices = num_vertices
        self.embed_dim = embed_dim
        self.learn_rate = learn_rate
        self.embeddings_device = embeddings_device

        # Initialize weights
        if initial_embeddings is None:
            # The discriminator's probability estimate is
            #   D = 1 - exp(-sum_j(prod_i(d_ij)))
            # Treating the embeddings as i.i.d. random variables:
            #   D = 1 - exp( -embed_dim * d^motif_size )
            #   log(d) = log( -log(1-D) / embed_dim ) / motif_size
            # We initialize the embeddings in log-space so that the
            # discriminator's initial probability estimates have mean 0.5.
            mean = ( -math.log(1-0.5) / embed_dim ) ** (1/motif_size)
            radius = ( -math.log(1-0.75) / embed_dim ) ** (1/motif_size) - mean
            init = lbann.UniformInitializer(min=mean-radius, max=mean+radius)
        else:
            init = lbann.ValueInitializer(values=str_list(np.nditer(initial_embeddings)))
        self.embedding_weights = lbann.Weights(
            initializer=init,
            name='discriminator_embeddings',
        )

    def get_embeddings(self, indices):
        embeddings = lbann.DistEmbedding(
            indices,
            weights=self.embedding_weights,
            num_embeddings=self.num_vertices,
            embedding_dim=self.embed_dim,
            sparse_sgd=True,
            learning_rate=self.learn_rate,
            device=self.embeddings_device
        )

        # Force embeddings to be positive
        # Note: Propagate gradients even when embeddings are negative
        epsilon = 0.1
        embeddings = lbann.Sum(
            embeddings,
            lbann.Relu(lbann.Negative(lbann.StopGradient(embeddings))),
            lbann.Constant(value=epsilon, hint_layer=embeddings),
        )
        return embeddings

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
