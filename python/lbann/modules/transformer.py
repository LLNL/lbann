"""Neural network modules for transformer models."""
import math
import lbann
from .base import Module, FullyConnectedModule
from lbann.util import make_iterable


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
        name (str): Default name is in the form
            'multiheadattention<index>'.

    """

    global_count = 0  # Static counter, used for default names

    def __init__(self, embed_dim, num_heads, name=None):
        super().__init__()
        MultiheadAttention.global_count += 1
        self.instance = 0
        assert embed_dim % num_heads == 0, 'embed_dim must be divisible by num_heads'
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Module name
        self.name = name
        if not self.name:
            self.name = f'multiheadattention{MultiheadAttention.global_count}'

        # Weights for fully-connected layers
        self.query_weights = [
            lbann.Weights(initializer=lbann.GlorotNormalInitializer(),
                          name=f'{self.name}_query_matrix'),
            lbann.Weights(initializer=lbann.ConstantInitializer(value=0),
                          name=f'{self.name}_query_bias'),
        ]
        self.key_weights = [
            lbann.Weights(initializer=lbann.GlorotNormalInitializer(),
                          name=f'{self.name}_key_matrix'),
            lbann.Weights(initializer=lbann.ConstantInitializer(value=0),
                          name=f'{self.name}_key_bias'),
        ]
        self.value_weights = [
            lbann.Weights(initializer=lbann.GlorotNormalInitializer(),
                          name=f'{self.name}_value_matrix'),
            lbann.Weights(initializer=lbann.ConstantInitializer(value=0),
                          name=f'{self.name}_value_bias'),
        ]
        self.output_weights = [
            lbann.Weights(initializer=lbann.GlorotNormalInitializer(),
                          name=f'{self.name}_output_matrix'),
            lbann.Weights(initializer=lbann.ConstantInitializer(value=0),
                          name=f'{self.name}_output_bias'),
        ]

    def forward(self, queries, keys, values, mask=None):
        """Apply multi-head attention.

        The input and output tensors are interpreted as sequences of
        vectors, where the first tensor dimension is the sequence
        dimension.

        Args:
            queries (lbann.Layer): Sequence of query vectors.
            keys (lbann.Layer): Sequence of key vectors.
            values (lbann.Layer): Sequence of value vectors.
            mask (lbann.Layer, optional): Additive attention mask. If
                the (i,j) entry is very negative (e.g. -1e9), then the
                ith query does not attend to the jth key/value pair.

        Returns:
            lbann.Layer: Sequence of output vectors. The sequence
                length is the same as `queries`.

        """
        self.instance += 1
        name = f'{self.name}_instance{self.instance}'

        # Apply fully-connected layers to input sequences
        queries_fc = lbann.ChannelwiseFullyConnected(
            queries,
            weights=self.query_weights,
            output_channel_dims=[self.embed_dim],
            name=f'{name}_queries_fc',
        )
        keys_fc = lbann.ChannelwiseFullyConnected(
            keys,
            weights=self.key_weights,
            output_channel_dims=[self.embed_dim],
            name=f'{name}_keys_fc',
        )
        values_fc = lbann.ChannelwiseFullyConnected(
            values,
            weights=self.value_weights,
            output_channel_dims=[self.embed_dim],
            name=f'{name}_values_fc',
        )

        # Slice embedding vectors for each head
        slice_points = [self.head_dim * i for i in range(self.num_heads + 1)]
        queries_slice = lbann.Slice(
            queries_fc,
            axis=1,
            slice_points=slice_points,
            name=f'{name}_queries_slice',
        )
        keys_slice = lbann.Slice(
            keys_fc,
            axis=1,
            slice_points=slice_points,
            name=f'{name}_keys_slice',
        )
        values_slice = lbann.Slice(
            values_fc,
            axis=1,
            slice_points=slice_points,
            name=f'{name}_values_slice',
        )

        # Compute scaled dot-product attention for each head
        attentions = []
        for head in range(self.num_heads):
            head_name = f'{name}_head{head}'

            # Attention inputs
            q = lbann.Identity(queries_slice)
            k = lbann.Identity(keys_slice)
            v = lbann.Identity(values_slice)

            # Multiply queries and keys
            # Note: num_queries x num_keys
            y = lbann.MatMul(
                q,
                k,
                transpose_b=True,
                name=f'{head_name}_matmul',
            )
            y = lbann.Scale(y,
                            constant=1 / math.sqrt(self.head_dim),
                            name=f'{head_name}_scale')
            if mask:
                y = lbann.Add(y, mask, name=f'{head_name}_mask')
            y = lbann.ChannelwiseSoftmax(y, name=f'{head_name}_softmax')

            # Attention output
            # Note: num_queries x head_dim
            attentions.append(lbann.MatMul(y, v, name=head_name))

        # Concatenate heads and apply fully-connected layer
        attentions = lbann.Concatenation(attentions,
                                         axis=1,
                                         name=f'{name}_heads_concat')

        outputs_fc = lbann.ChannelwiseFullyConnected(
            attentions,
            weights=self.output_weights,
            output_channel_dims=[self.embed_dim],
            name=f'{name}',
        )
        return outputs_fc
