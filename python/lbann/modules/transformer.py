"""Neural network modules for transformer models."""

from lbann.utils import make_iterable, str_list, make_nd_array
from .base import Module, FullyConnectedModule


class MultiheadAttention(Module):

    def __init__(self, embed_dim, num_heads,
                 bias=True,
                 name=None):
        super().__init__()
        self.instance = 0
        self.name = (name
                     if name
                     else f'multiheadattention{MultiheadAttention.global_count}')

        assert embed_dim % num_heads == 0, 'embed_dim must be divisible by num_heads'
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads

        self.query_fc = FullyConnectedModule(
            self.embed_dim,
            bias=bias,
            name=f'{self.name}_query_fc'
        )
        self.key_fc = FullyConnectedModule(
            self.embed_dim,
            bias=bias,
            name=f'{self.name}_key_fc'
        )
        self.value_fc = FullyConnectedModule(
            self.embed_dim,
            bias=bias,
            name=f'{self.name}_value_fc'
        )
        self.output_fc = FullyConnectedModule(
            self.embed_dim,
            bias=bias,
            name=f'{self.name}_value_fc'
        )

    def project_input_sequence(self, proj, input_seq):

        # Apply projection to input sequence
        input_seq = make_iterable(input_seq)
        seq_length = len(input_seq)
        proj_seq = [proj(x) for x in input_seq]

        # Slice and rearrange into one 2D tensor per head
        slice_points = str_list(self.head_dim * i
                                for i in range(self.num_heads+1))
        proj_seq = [lbann.Slice(proj(x), slice_points=slice_points)
                    for x in input_seq]
        proj_array = make_nd_list(self.num_heads, seq_length)
        for seq_pos, head in itertools.product(range(seq_length),
                                               range(num_heads)):
            proj_array[head][seq_pos] = lbann.Resize(proj_seq[seq_pos],
                                                     dims='1 -1')
        return [lbann.Concatenation(ps) for ps in proj_array]

    def forward(self, queries, keys, values):

        # Apply FC layers to input sequences
        num_queries = len(queries)
        queries_proj = self.project_input_sequence(self.query_fc, queries)
        keys_proj = self.project_input_sequence(self.key_fc, keys)
        values_proj = self.project_input_sequence(self.value_fc, values)

        # Compute scaled dot-product attention for each head
        attentions = []
        for head in range(self.num_heads):

            # Attention inputs
            q = queries_proj[head]
            k = keys_proj[head]
            v = values_proj[head]

            # Multiply queries and keys
            y = lbann.MatMult(q, k, transpose_b=True)
            scale = 1 / math.sqrt(self.head_dim)
            y = lbann.WeightedSum(y, scaling_factors=str(scale))

            # Row-wise softmax
            y = lbann.Slice(y, slice_points=str_list(range(num_queries+1)))
            y = [lbann.Softmax(y) for _ in range(num_queries)]
            y = lbann.Concatenation(y)

            # Attention output
            # num_queries x head_dim
            attentions.append(lbann.MatMul(y, v))

        # Concatenate attention heads and apply FC layer
        attentions = lbann.Concatenation(attentions, axis=1)
        attentions = lbann.Slice(attentions,
                                 slice_points=str_list(range(num_queries)+1))
        attentions = [lbann.Identity(attentions) for _ in range(num_queries)]
        return [self.output_fc(z) for z in attentions]
