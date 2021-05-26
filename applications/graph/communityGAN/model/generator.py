import math
import numpy as np
import lbann
import lbann.modules

from util import str_list

class Generator(lbann.modules.Module):

    def __init__(
            self,
            num_vertices,
            embed_dim,
            learn_rate,
            embeddings_device='CPU',
    ):
        super().__init__()
        self.num_vertices = num_vertices
        self.embed_dim = embed_dim
        self.learn_rate = learn_rate
        self.embeddings_device = embeddings_device

        # Initialize weights
        # Note: The generator's confidence for adding vertex v to a
        # fake motif is
        #   G(v) = G'(v) / sum_w(G'(w))
        #   G'(v) = 1 - exp(-sum_j(prod_i(g_ij) * g_vj))
        # Treating the embeddings as i.i.d. random variables:
        #   G' = 1 - exp( -embed_dim * g^(motif_size+1) )
        #   log(g) = log( -log(1-G') / embed_dim ) / (motif_size+1)
        # We initialize the embeddings in log-space so that the
        # numerator G' has mean 0.5 in the first generator step (i.e.
        # motif_size=1).
        mean = math.log( -math.log(1-0.5) / embed_dim ) / (1+1)
        radius = math.log( -math.log(1-0.75) / embed_dim ) / (1+1) - mean
        self.log_embedding_weights = lbann.Weights(
            initializer=lbann.UniformInitializer(
                min=mean-radius, max=mean+radius),
            name='generator_log_embeddings',
        )

        # Initialize caches for helper functions
        self.tril_ones_cache = {}
        self.iota_cache = {}

    def forward(
            self,
            num_candidates,
            candidate_indices,
            motif_size,
    ):

        # Get log of embeddings for candidate vertices
        candidate_log_embeddings = self._get_log_embeddings(candidate_indices)

        # Initialize motif with first candidate vertex
        motif_indices = [
            lbann.Slice(candidate_indices, slice_points=str_list([0,1])),
        ]
        motif_mask = lbann.Add(
            lbann.Concatenation(
                lbann.Constant(value=1, num_neurons=str(1)),
                lbann.Constant(value=0, num_neurons=str(num_candidates-1)),
            ),
            lbann.Less(
                candidate_indices,
                lbann.Constant(value=0, num_neurons=str(num_candidates)),
            ),
        )
        motif_mask = lbann.Reshape(motif_mask, dims=str_list([num_candidates,1]))

        # Generate motif
        log_probs = []
        for _ in range(1, motif_size):
            choice_index, motif_mask, log_prob = self._expand_motif(
                num_candidates,
                candidate_indices,
                candidate_log_embeddings,
                motif_mask,
            )
            motif_indices.append(choice_index)
            log_probs.append(log_prob)
        motif_indices = lbann.Concatenation(motif_indices)
        log_prob = lbann.Sum(log_probs)
        prob = lbann.Exp(log_prob)

        return motif_indices, prob, log_prob

    def _get_log_embeddings(self, indices):
        return lbann.DistEmbedding(
            indices,
            weights=self.log_embedding_weights,
            num_embeddings=self.num_vertices,
            embedding_dim=self.embed_dim,
            sparse_sgd=True,
            learning_rate=self.learn_rate,
            device=self.embeddings_device,
        )

    def _expand_motif(
            self,
            num_candidates,
            candidate_indices,
            candidate_log_embeddings,
            motif_mask,
    ):

        # -sum_j(prod_i(g_ij)*g_vj) = -sum_j(exp(sum_i(log(g_ij))+g_vj))
        x = lbann.MatMul(
            motif_mask,
            candidate_log_embeddings,
            transpose_a=True,
        )
        x = lbann.Add(
            candidate_log_embeddings,
            lbann.Tessellate(x, hint_layer=candidate_log_embeddings),
        )
        x = lbann.Exp(x)
        x = lbann.MatMul(
            x,
            lbann.Constant(value=1, num_neurons=str_list([self.embed_dim, 1])),
        )
        x = lbann.Negative(x)

        # G = (1-exp(-sum_j(prod_i(g_ij)))) / sum(1-exp(-sum_j(prod_i(g_ij))))
        # log(G) = log_softmax(log1p(-exp( -sum_j(prod_i(g_ij)*g_vj) )))
        x = lbann.Log1p(lbann.Negative(lbann.Exp(x)))
        x = lbann.Add(
            x,
            self._bool_to_value_mask(motif_mask, -1e5),
        )
        x = lbann.Reshape(x, dims='-1')
        log_probs = lbann.LogSoftmax(x)

        # Pick best vertex
        # Note: Pick randomly with distribution from G
        probs = lbann.Exp(log_probs)
        probs_cumsum = lbann.MatMul(
            self._tril_ones((num_candidates, num_candidates), -1),
            lbann.Reshape(probs, dims='-1 1'),
        )
        probs_cumsum = lbann.Reshape(probs_cumsum, dims='-1', device='CPU')
        rand = lbann.Uniform(min=0, max=1, neuron_dims='1', device='CPU')
        choice = lbann.Argmax(
            lbann.Multiply(
                lbann.LessEqual(
                    probs_cumsum,
                    lbann.Tessellate(rand, hint_layer=probs_cumsum, device='CPU'),
                    device='CPU',
                ),
                self._iota(num_candidates),
                device='CPU',
            ),
            device='CPU',
        )
        choice_index = lbann.Gather(candidate_indices, choice)

        # Loss function
        log_prob = lbann.Gather(log_probs, choice)

        # Add choice to motif
        choice_onehot = lbann.OneHot(choice, size=num_candidates)
        choice_onehot = lbann.Reshape(choice_onehot, hint_layer=motif_mask)
        motif_mask = lbann.Add(motif_mask, choice_onehot)
        return choice_index, motif_mask, log_prob

    def _bool_to_value_mask(self, bool_mask, value):
        value_mask = lbann.Constant(value=value, hint_layer=bool_mask)
        value_mask = lbann.Multiply(value_mask, bool_mask)
        return value_mask

    def _tril_ones(self, dims, k):
        if (dims, k) not in self.tril_ones_cache:
            vals = np.tril(np.full(dims, 1, dtype=int), k=k)
            w = lbann.Weights(
                initializer=lbann.ValueInitializer(values=str_list(np.nditer(vals))),
                optimizer=lbann.NoOptimizer(),
            )
            self.tril_ones_cache[(dims, k)] = lbann.WeightsLayer(
                dims=str_list(dims),
                weights=w,
            )
        return self.tril_ones_cache[(dims, k)]

    def _iota(self, size):
        """Generated on CPU"""
        if size not in self.iota_cache:
            w = lbann.Weights(
                initializer=lbann.ValueInitializer(values=str_list(range(size))),
                optimizer=lbann.NoOptimizer(),
            )
            self.iota_cache[size] = lbann.WeightsLayer(
                dims=str(size),
                weights=w,
                device='CPU',
            )
        return self.iota_cache[size]
