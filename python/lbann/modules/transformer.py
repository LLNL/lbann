"""Neural network modules for transformer models."""
import math
import lbann
from .base import Module, FullyConnectedModule
from lbann.util import make_iterable, str_list

class MultiheadAttention(Module):
    """Parallel instances of scaled dot-product attention.

    See:

    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion
    Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin.
    "Attention is all you need." In Advances in Neural Information
    Processing Systems, pp. 5998-6008. 2017.

    Args:
        embed_dim (int): Size of representation space.
        num_heads (int): Number of parallel attention instances. Must
            evenly divide `embed_dim`.
        bias (bool): Whether to apply bias in internal fully-connected
            layers.
        name (str): Default name is in the form
            'multiheadattention<index>'.

    """

    global_count = 0  # Static counter, used for default names

    def __init__(self,
                 embed_dim,
                 num_heads,
                 bias=True,
                 name=None):
        super().__init__()
        MultiheadAttention.global_count += 1
        self.instance = 0
        self.name = (
            name
            if name
            else f'multiheadattention{MultiheadAttention.global_count}'
        )
        assert embed_dim % num_heads == 0, 'embed_dim must be divisible by num_heads'
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Fully-connected modules
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
            name=f'{self.name}_output_fc'
        )

    def _project_input_sequence(self, proj, input_seq):
        """Apply projection operator to sequence of vectors.

        The projection operator is applied independently to each
        vector in the sequence. The results are rearranged and split
        into `num_heads` matrices, each of which has dimensions of
        `sequence_length x head_dim matrix`.

        Args:
            proj (lbann.modules.FullyConnectedModule): Projection
                operator.
            input_seq (Iterable of lbann.Layer): Sequence of input
                vectors.

        Returns:
            list of lbann.Layer: Matrix for each attention head.

        """

        # Make sure input sequence is valid
        input_seq = make_iterable(input_seq)
        assert len(input_seq) > 0, 'input sequence is empty'

        # Apply projection to each vector in input sequence
        proj_seq = []
        for i in range(len(input_seq)):
            x = proj(input_seq[i])
            name = x.name
            x = lbann.Reshape(
                x,
                dims='1 -1',
                name=f'{name}_reshape'
            )
            proj_seq.append(x)

        # Rearrange projections into one matrix per head
        proj_concat = lbann.Concatenation(
            proj_seq,
            axis=0,
            name=f'{name}_concat'
        )
        proj_slice = lbann.Slice(
            proj_concat,
            axis=1,
            slice_points=str_list(
                self.head_dim * i
                for i in range(self.num_heads+1)
            ),
            name=f'{name}_slice'
        )
        return [
            lbann.Identity(proj_slice, name=f'{name}_head{i}')
            for i in range(self.num_heads)
        ]

    def forward(self, queries, keys, values, mask=None):
        """Apply multi-head attention.

        Args:
            queries (Iterable of lbann.Layer): Sequence of query
                vectors.
            keys (Iterable of lbann.Layer): Sequence of key vectors.
            values (Iterable of lbann.Layer): Sequence of value
                vectors. Sequence length be same as `keys`.
            mask (lbann.Layer, optional): Additive attention mask. If
                the (i,j) entry is very negative (e.g. -1e9), then the
                ith query does not attend to the jth key/value pair.

        Returns:
            list of lbann.Layer: Sequence of output vectors. Sequence
                length is same as `queries`.

        """
        self.instance += 1

        # Apply FC layers to input sequences
        num_queries = len(queries)
        queries_proj = self._project_input_sequence(self.query_fc, queries)
        keys_proj = self._project_input_sequence(self.key_fc, keys)
        values_proj = self._project_input_sequence(self.value_fc, values)

        # Compute scaled dot-product attention for each head
        attentions = []
        for head in range(self.num_heads):
            name = f'{self.name}_instance{self.instance}_head{head}'

            # Attention inputs
            q = queries_proj[head]
            k = keys_proj[head]
            v = values_proj[head]

            # Multiply queries and keys
            # Note: num_queries x num_keys
            y = lbann.MatMul(
                q, k,
                transpose_b=True,
                name=f'{name}_matmul',
            )
            y = lbann.WeightedSum(
                y,
                scaling_factors=str(1 / math.sqrt(self.head_dim)),
                name=f'{name}_scale',
            )

            # Apply additive mask if provided
            if mask:
                y = lbann.Add(y, mask, name=f'{name}_mask')

            # Row-wise softmax
            # Note: cuDNN's softmax implementation requires that y and
            # dy have the same stride. However, the error signal
            # emitted by the concatenation layer are not fully-packed.
            # To get around this problem, we insert a dummy layer
            # after softmax to ensure error signals are fully-packed.
            y = lbann.Slice(
                y,
                axis=0,
                slice_points=str_list(range(num_queries+1)),
                name=f'{name}_softmax_slice'
            )
            y = [
                lbann.Softmax(y, name=f'{name}_softmax{i}')
                for i in range(num_queries)
            ]
            y = [lbann.Relu(yi, name=f'{yi.name}_dummy') for yi in y]
            y = lbann.Concatenation(y, axis=0, name=f'{name}_softmax')

            # Attention output
            # Note: num_queries x head_dim
            attentions.append(lbann.MatMul(y, v, name=name))

        # Concatenate attention heads and apply FC layer
        name = f'{self.name}_instance{self.instance}'
        attentions = lbann.Concatenation(
            attentions,
            axis=1,
            name=f'{name}_concat'
        )
        attentions = lbann.Slice(
            attentions,
            axis=0,
            slice_points=str_list(range(num_queries+1)),
            name=f'{name}_slice'
        )
        attentions = [
            lbann.Identity(attentions, name=f'{name}_seq{i}')
            for i in range(num_queries)
        ]
        return [self.output_fc(z) for z in attentions]
