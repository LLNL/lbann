import math
import lbann
import lbann.modules

from util import str_list

class GreedyGenerator(lbann.modules.Module):

    def __init__(
            self,
            num_vertices,
            embed_dim,
            learn_rate,
    ):
        super().__init__()
        self.num_vertices = num_vertices
        self.embed_dim = embed_dim
        self.learn_rate = learn_rate

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
        motif_mask = lbann.Concatenation(
            lbann.Constant(value=1, num_neurons=str_list([1,1])),
            lbann.Constant(value=0, num_neurons=str_list([num_candidates-1,1])),
        )

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
        ### @todo Use gather instead of one-hot vector
        ### @todo Consider choosing propabilistically
        choice = lbann.Argmax(log_probs, device='CPU')
        choice_onehot = lbann.OneHot(choice, size=num_candidates)
        choice_onehot = lbann.Reshape(choice_onehot, hint_layer=motif_mask)
        choice_index = lbann.MatMul(
            lbann.Reshape(candidate_indices, dims='1 -1'),
            choice_onehot,
        )
        choice_index = lbann.Reshape(choice_index, dims='1')

        # Loss function
        log_prob = lbann.MatMul(
            lbann.Reshape(log_probs, dims='1 -1'),
            choice_onehot,
        )
        log_prob = lbann.Reshape(log_prob, dims='1')

        # Add choice to motif
        motif_mask = lbann.Add(motif_mask, choice_onehot)
        return choice_index, motif_mask, log_prob

    def _bool_to_value_mask(self, bool_mask, value):
        value_mask = lbann.Constant(value=value, hint_layer=bool_mask)
        value_mask = lbann.Multiply(value_mask, bool_mask)
        return value_mask
