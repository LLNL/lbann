"""Neural network modules for multi-head attention."""
import math
from typing import Dict, Optional
import lbann
from ..base import Module
from lbann.modules.transformer.encoding import SequenceEncoding


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
        self_attention (bool): If True, performs self-attention on the same
            tensor.
        batch_heads (bool): If True, batches all head computations into tensor
            contraction operations.
        dropout (float): Dropout probability applied before attention output.
        subgraph_branches (int): How many subgraph-parallel branches to divide
            multi-head attention to.
        bias (Layer): Additive attention bias to apply on the attention
            probability matrix before softmax. If None, does not apply.
        positional_encoding (SequenceEncoding): An optional positional encoding
            object that may apply on each input.
        name (str): Default name is in the form
            'multiheadattention<index>'.

    """

    global_count = 0  # Static counter, used for default names

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 self_attention: bool = False,
                 batch_heads: bool = True,
                 dropout: float = 0.0,
                 subgraph_branches: int = 0,
                 bias: Optional[lbann.Layer] = None,
                 positional_encoding: Optional[SequenceEncoding] = None,
                 name: str = None):
        super().__init__()
        MultiheadAttention.global_count += 1
        self.instance = 0
        assert embed_dim % num_heads == 0, 'embed_dim must be divisible by num_heads'
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.bias = bias
        self.positional_encoding = positional_encoding

        # Self-attention is a special case in which we can stack
        # query/key/value weights
        self.self_attention = self_attention

        # Mode that runs each head separately and concatenates the results
        self.subgraph_branches = subgraph_branches
        if subgraph_branches > 0:
            self.separate_heads = True
            if self.num_heads % subgraph_branches != 0:
                raise ValueError('Number of heads should be divisible by '
                                 'parallel attention head branches')
        else:
            self.separate_heads = not batch_heads

        # Module name
        self.name = name
        if not self.name:
            self.name = f'multiheadattention{MultiheadAttention.global_count}'

        # Weights for fully-connected layers
        if self_attention:
            self.qkv_weights = [
                lbann.Weights(initializer=lbann.GlorotNormalInitializer(),
                              name=f'{self.name}_qkv_matrix'),
                lbann.Weights(initializer=lbann.ConstantInitializer(value=0),
                              name=f'{self.name}_qkv_bias'),
            ]
        else:
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

    def forward(self,
                queries,
                keys,
                values,
                mask=None,
                seqlen=None,
                **extra_kwargs):
        """Apply multi-head attention.

        The input and output tensors are interpreted as sequences of
        vectors, where the first tensor dimension is the sequence
        dimension.

        Args:
            queries (lbann.Layer): Sequence of query vectors.
            keys (lbann.Layer): Sequence of key vectors.
            values (lbann.Layer): Sequence of value vectors.
            mask (lbann.Layer or List[lbann.Layer], optional): Additive
                attention mask or masks, if attention-head parallelism is
                enabled. If the (i,j) entry is very negative (e.g. -1e9), then
                the ith query does not attend to the jth key/value pair.
            seqlen (int): Optional sequence length of the current set of
                sequences. If not given, max sequence length is assumed.

        Returns:
            lbann.Layer: Sequence of output vectors. The sequence
                length is the same as `queries`.

        """
        self.instance += 1
        name = f'{self.name}_instance{self.instance}'

        if self.self_attention and queries is keys and keys is values:
            # If self-attention, multiply with stacked matrix
            qkv_fc = lbann.ChannelwiseFullyConnected(
                queries,
                weights=self.qkv_weights,
                output_channel_dims=[self.embed_dim * 3],
                name=f'{name}_qkv_fc',
                bias=True,
                transpose=False,
                **extra_kwargs)

            # Unstack
            qkv_slice = lbann.Slice(qkv_fc,
                                    axis=1,
                                    slice_points=[
                                        0, self.embed_dim, 2 * self.embed_dim,
                                        3 * self.embed_dim
                                    ],
                                    **extra_kwargs)
            queries_fc = lbann.Identity(qkv_slice, **extra_kwargs)
            keys_fc = lbann.Identity(qkv_slice, **extra_kwargs)
            values_fc = lbann.Identity(qkv_slice, **extra_kwargs)
        else:
            # Otherwise, apply fully-connected layers to input sequences separately
            queries_fc = lbann.ChannelwiseFullyConnected(
                queries,
                weights=self.query_weights,
                output_channel_dims=[self.embed_dim],
                name=f'{name}_queries_fc',
                **extra_kwargs,
            )
            keys_fc = lbann.ChannelwiseFullyConnected(
                keys,
                weights=self.key_weights,
                output_channel_dims=[self.embed_dim],
                name=f'{name}_keys_fc',
                **extra_kwargs,
            )
            values_fc = lbann.ChannelwiseFullyConnected(
                values,
                weights=self.value_weights,
                output_channel_dims=[self.embed_dim],
                name=f'{name}_values_fc',
                **extra_kwargs,
            )

        if self.positional_encoding is not None:
            queries_fc, keys_fc, values_fc = self.positional_encoding.apply_layer(
                queries_fc, keys_fc, values_fc, seqlen, **extra_kwargs)

        if self.separate_heads:
            attentions = self.dot_product_attn_separate_heads(
                name, queries_fc, keys_fc, values_fc, mask, **extra_kwargs)
        else:
            attentions = self.dot_product_attn_batched(name, queries_fc,
                                                       keys_fc, values_fc,
                                                       mask, **extra_kwargs)

        outputs_fc = lbann.ChannelwiseFullyConnected(
            attentions,
            weights=self.output_weights,
            output_channel_dims=[self.embed_dim],
            name=f'{name}',
            **extra_kwargs,
        )
        return outputs_fc

    def dot_product_attn_batched(self, name, queries_fc, keys_fc, values_fc,
                                 mask, **extra_kwargs):
        head_name = f'{name}_all_heads'
        queries_fc = lbann.Scale(
            queries_fc,
            constant=1 / math.sqrt(self.head_dim),
            name=f'{head_name}_scale',
            **extra_kwargs,
        )

        # Dimension key:
        #   * S = Sequence length
        #   * H = Number of heads
        #   * E = Embedding dimension
        #   * P = Head size

        # SxE -> HxPxS
        q_headsfirst = lbann.TensorPermute(queries_fc,
                                           axes=(1, 0),
                                           **extra_kwargs)
        q_headsfirst = lbann.Reshape(q_headsfirst,
                                     dims=(self.num_heads, self.head_dim, -1),
                                     **extra_kwargs)
        k_headsfirst = lbann.TensorPermute(keys_fc,
                                           axes=(1, 0),
                                           **extra_kwargs)
        k_headsfirst = lbann.Reshape(k_headsfirst,
                                     dims=(self.num_heads, self.head_dim, -1),
                                     **extra_kwargs)
        v_headsfirst = lbann.TensorPermute(values_fc,
                                           axes=(1, 0),
                                           **extra_kwargs)
        v_headsfirst = lbann.Reshape(v_headsfirst,
                                     dims=(self.num_heads, self.head_dim, -1),
                                     **extra_kwargs)

        # HxPxS -> HxSxS
        y = lbann.MatMul(
            q_headsfirst,
            k_headsfirst,
            transpose_a=True,
            transpose_b=False,
            name=f'{head_name}_matmul',
            **extra_kwargs,
        )

        if mask:
            y = lbann.Add(y, mask, name=f'{head_name}_mask', **extra_kwargs)

        if self.bias:
            y = lbann.Add(y,
                          self.bias,
                          name=f'{head_name}_attnbias',
                          **extra_kwargs)

        y = lbann.ChannelwiseSoftmax(y,
                                     dim=-1,
                                     single_dim_mode=True,
                                     name=f'{head_name}_softmax',
                                     **extra_kwargs)

        if self.dropout > 0:
            y = lbann.Dropout(
                y,
                keep_prob=1 - self.dropout,
                name=f'{head_name}_drop',
                **extra_kwargs,
            )

        # Attention output as batched matrix multiplication
        # HxSxS * HxSxP -> HxSxP
        attentions = lbann.MatMul(y,
                                  v_headsfirst,
                                  transpose_b=True,
                                  name=head_name,
                                  **extra_kwargs)

        # HxSxP -> SxE
        attentions = lbann.TensorPermute(attentions,
                                         axes=(1, 0, 2),
                                         **extra_kwargs)
        attentions = lbann.Reshape(attentions,
                                   dims=(-1, self.embed_dim),
                                   **extra_kwargs)
        return attentions

    def _get_subgraph(self, tag_id: int = 0) -> Dict[str, int]:
        """
        Returns a parallel strategy based on the attention head subgraph
        parallelism configuration and the requested grid tag.

        :param tag_id: The preferred grid tag to use.
        :return: A dictionary mapping to the subgraph if subgraph parallelism
                 is requested, or None otherwise.
        """
        if self.subgraph_branches == 0:
            return None
        return dict(grid_tag=tag_id)

    def dot_product_attn_separate_heads(self, name, queries_fc, keys_fc,
                                        values_fc, mask, **extra_kwargs):
        # Slice embedding vectors for each head
        slice_points = [self.head_dim * i for i in range(self.num_heads + 1)]
        queries_slice = lbann.Slice(
            queries_fc,
            axis=1,
            slice_points=slice_points,
            name=f'{name}_queries_slice',
            parallel_strategy=self._get_subgraph(),
            **extra_kwargs,
        )
        keys_slice = lbann.Slice(
            keys_fc,
            axis=1,
            slice_points=slice_points,
            name=f'{name}_keys_slice',
            parallel_strategy=self._get_subgraph(),
            **extra_kwargs,
        )
        values_slice = lbann.Slice(
            values_fc,
            axis=1,
            slice_points=slice_points,
            name=f'{name}_values_slice',
            parallel_strategy=self._get_subgraph(),
            **extra_kwargs,
        )

        if self.subgraph_branches > 0 and mask is not None:
            assert (isinstance(mask, list)
                    and len(mask) == self.subgraph_branches)

        # Compute scaled dot-product attention for each head
        attentions = []
        tag = 0
        for head in range(self.num_heads):
            head_name = f'{name}_head{head}'

            # Increment tag when head subgraph branch is fully populated
            if (self.subgraph_branches > 0 and head %
                (self.num_heads // self.subgraph_branches) == 0):
                tag += 1

            # Attention inputs
            q = lbann.Identity(queries_slice,
                               parallel_strategy=self._get_subgraph(tag),
                               **extra_kwargs)
            k = lbann.Identity(keys_slice,
                               parallel_strategy=self._get_subgraph(tag),
                               **extra_kwargs)
            v = lbann.Identity(values_slice,
                               parallel_strategy=self._get_subgraph(tag),
                               **extra_kwargs)

            # Multiply queries and keys
            # Note: num_queries x num_keys
            y = lbann.MatMul(
                q,
                k,
                transpose_b=True,
                name=f'{head_name}_matmul',
                **extra_kwargs,
            )
            y = lbann.Scale(y,
                            constant=1 / math.sqrt(self.head_dim),
                            name=f'{head_name}_scale',
                            **extra_kwargs)
            if mask:
                if self.subgraph_branches > 0:
                    y = lbann.Add(y,
                                  mask[tag - 1],
                                  name=f'{head_name}_mask',
                                  **extra_kwargs)
                else:
                    y = lbann.Add(y,
                                  mask,
                                  name=f'{head_name}_mask',
                                  **extra_kwargs)

            if self.bias:
                y = lbann.Add(y,
                              self.bias,
                              name=f'{head_name}_attnbias',
                              **extra_kwargs)

            y = lbann.ChannelwiseSoftmax(y,
                                         name=f'{head_name}_softmax',
                                         **extra_kwargs)

            if self.dropout > 0:
                y = lbann.Dropout(
                    y,
                    keep_prob=1 - self.dropout,
                    name=f'{head_name}_drop',
                    **extra_kwargs,
                )

            # Attention output
            # Note: num_queries x head_dim
            attentions.append(
                lbann.MatMul(y, v, name=head_name, **extra_kwargs))

        # Concatenate heads and apply fully-connected layer
        attentions = lbann.Concatenation(
            attentions,
            axis=1,
            name=f'{name}_heads_concat',
            parallel_strategy=self._get_subgraph(),
            **extra_kwargs)
        return attentions
