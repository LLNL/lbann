"""Basic Transformer model with multi-head self-attention.

See:

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion
Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. "Attention
is all you need." In Advances in Neural Information Processing
Systems, pp. 5998-6008. 2017.

"""
import math
import numpy as np
from typing import Optional

import lbann
import lbann.modules
from lbann.modules.transformer.encoding import SequenceEncoding
from lbann.util import make_iterable


class LayerNorm(lbann.modules.Module):
    """See https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html"""

    global_count = 0  # Static counter, used for default names

    def __init__(self, normalized_shape, name=None, builtin=True):
        super().__init__()
        LayerNorm.global_count += 1
        self.normalized_shape = make_iterable(normalized_shape)
        self.name = (name if name else f'layernorm{LayerNorm.global_count}')
        self.builtin = builtin

        # Initialize weights
        self.weight = lbann.Weights(
            initializer=lbann.ConstantInitializer(value=1),
            name=f'{self.name}_weight',
        )
        self.bias = lbann.Weights(
            initializer=lbann.ConstantInitializer(value=0),
            name=f'{self.name}_bias',
        )

    def forward(self, x, **extra_kwargs):
        if self.builtin:
            return lbann.LayerNorm(x,
                                   scale=True,
                                   bias=True,
                                   start_dim=-1,
                                   name=self.name,
                                   weights=[self.weight, self.bias],
                                   **extra_kwargs)

        # Normalization
        x = lbann.InstanceNorm(x, **extra_kwargs)

        # Affine transform
        s = lbann.WeightsLayer(
            weights=self.weight,
            dims=[1] + list(make_iterable(self.normalized_shape)),
            **extra_kwargs,
        )
        s = lbann.Tessellate(s, hint_layer=x, **extra_kwargs)
        b = lbann.WeightsLayer(
            weights=self.bias,
            dims=[1] + list(make_iterable(self.normalized_shape)),
            **extra_kwargs,
        )
        b = lbann.Tessellate(b, hint_layer=x, **extra_kwargs)
        x = lbann.Add(lbann.Multiply(s, x, **extra_kwargs), b, **extra_kwargs)
        return x


class TransformerEncoderLayer(lbann.modules.Module):
    """Building block for encoder in Transformer model.

    Comprised of multi-head attention and a fully-connected
    feedforward network, each with a residual connection.

    Args:
        embed_dim (int): Internal dimensionality of multi-head
            attention.
        num_heads (int): Number of attention heads.
        feedforward_dim (int): Internal dimensionality of
            fully-connected feedforward network.
        dropout (float): Dropout probability after multi-head attention.
        attn_dropout (float): Dropout probability during multi-head attention.
        pre_layernorm (bool): If True, performs layer normalization before
            applying attention operators.
        activation (Type[lbann.Layer]): Activation function to apply in
            feedforward network. Examples include ReLU or GELU.
        parallel_attention_heads (int): If positive, applies subgraph
            parallelism on attention heads.
        attention_bias (Layer): Additive attention bias to apply on the attention
            probability matrix before softmax. If None, does not apply.
        positional_encoding (SequenceEncoding): An optional positional encoding
            object that may apply on each input.
        attention_module (Type[Module]): Sets the internal attention
            (self-attention and cross attention) class. By default, uses
            Multi-Head Attention.
        name (str): Default name is in the form
            'transformerencoderlayer<index>'.

    """

    global_count = 0  # Static counter, used for default names

    def __init__(
        self,
        embed_dim,
        num_heads,
        feedforward_dim,
        dropout,
        attn_dropout,
        pre_layernorm=False,
        activation=lbann.Relu,
        parallel_attention_heads=0,
        attention_bias=None,
        positional_encoding: Optional[SequenceEncoding] = None,
        attention_module=lbann.modules.MultiheadAttention,
        name=None,
    ):
        TransformerEncoderLayer.global_count += 1
        self.instance = 0
        self.embed_dim = embed_dim
        self.feedforward_dim = feedforward_dim
        self.dropout_prob = dropout
        self.pre_layernorm = pre_layernorm
        self.activation = activation
        self.extra_ffn_args = {}
        self.extra_layer_args = {}

        # Module name
        self.name = name
        if not self.name:
            self.name = f'transformerencoderlayer{TransformerEncoderLayer.global_count}'

        # Layer modules
        self.attention = attention_module(
            self.embed_dim,
            num_heads,
            dropout=attn_dropout,
            self_attention=True,
            subgraph_branches=parallel_attention_heads,
            bias=attention_bias,
            positional_encoding=positional_encoding,
            name=f'{self.name}_attention')
        self.norm1 = LayerNorm(self.embed_dim, name=f'{self.name}_norm1')
        self.norm2 = LayerNorm(self.embed_dim, name=f'{self.name}_norm2')

        # Weights for fully-connected layers
        self.fc1_weights = [
            lbann.Weights(initializer=lbann.HeNormalInitializer(),
                          name=f'{self.name}_fc1_matrix'),
            lbann.Weights(initializer=lbann.ConstantInitializer(value=0),
                          name=f'{self.name}_fc1_bias'),
        ]
        self.fc2_weights = [
            lbann.Weights(initializer=lbann.GlorotNormalInitializer(),
                          name=f'{self.name}_fc2_matrix'),
            lbann.Weights(initializer=lbann.ConstantInitializer(value=0),
                          name=f'{self.name}_fc2_bias'),
        ]

    def forward(self, x, mask=None):
        """Apply Transformer encoder layer.

        Args:
            x (lbann.Layer): Sequence of input vectors.
            mask (lbann.Layer, optional): Attention mask.

        Returns:
            lbann.Layer: Sequence of output vectors.

        """
        self.instance += 1
        name = f'{self.name}_instance{self.instance}'

        if self.pre_layernorm:
            y = self.norm1(x, **self.extra_layer_args)
        else:
            y = x

        # Self-attention with residual connection
        y = self.attention(y, y, y, mask=mask, **self.extra_layer_args)
        if self.dropout_prob > 0:
            y = lbann.Dropout(
                y,
                keep_prob=1 - self.dropout_prob,
                name=f'{name}_drop1',
                **self.extra_layer_args,
            )
        z = lbann.Sum(x, y, name=f'{name}_sum1', **self.extra_layer_args)
        if not self.pre_layernorm:
            z = self.norm1(z, **self.extra_layer_args)
        x = z

        # Feedforward network with residual connection
        if self.pre_layernorm:
            y = self.norm2(z, **self.extra_layer_args)
        else:
            y = x

        y = lbann.ChannelwiseFullyConnected(
            y,
            weights=self.fc1_weights,
            output_channel_dims=[self.feedforward_dim],
            name=f'{name}_fc1',
            **self.extra_layer_args,
            **self.extra_ffn_args,
        )
        y = self.activation(y,
                            name=f'{name}_ffn_act',
                            **self.extra_layer_args,
                            **self.extra_ffn_args)
        if self.dropout_prob > 0:
            y = lbann.Dropout(
                y,
                keep_prob=1 - self.dropout_prob,
                name=f'{name}_drop2',
                **self.extra_layer_args,
                **self.extra_ffn_args,
            )
        y = lbann.ChannelwiseFullyConnected(
            y,
            weights=self.fc2_weights,
            output_channel_dims=[self.embed_dim],
            name=f'{name}_fc2',
            **self.extra_layer_args,
            **self.extra_ffn_args,
        )
        if self.dropout_prob > 0:
            y = lbann.Dropout(
                y,
                keep_prob=1 - self.dropout_prob,
                name=f'{name}_drop3',
                **self.extra_layer_args,
                **self.extra_ffn_args,
            )
        z = lbann.Sum(x, y, name=f'{name}_sum2', **self.extra_layer_args)
        if not self.pre_layernorm:
            z = self.norm2(z, **self.extra_layer_args)
        return z


class TransformerDecoderLayer(lbann.modules.Module):
    """Building block for decoder in Transformer model.

    Comprised of two multi-head attention modules and a
    fully-connected feedforward network, each with a residual
    connection.

    Args:
        embed_dim (int): Internal dimensionality of multi-head
            attention.
        num_heads (int): Number of attention heads.
        feedforward_dim (int): Internal dimensionality of
            fully-connected feedforward network.
        dropout (float): Dropout probability after multi-head attention.
        attn_dropout (float): Dropout probability during multi-head attention.
        pre_layernorm (bool): If True, performs layer normalization before
            applying attention operators.
        activation (Type[lbann.Layer]): Activation function to apply in
            feedforward network. Examples include ReLU or GELU.
        parallel_attention_heads (int): If positive, applies subgraph
            parallelism on attention heads.
        attention_bias (Layer): Additive attention bias to apply on the attention
            probability matrix before softmax. If None, does not apply.
        positional_encoding (SequenceEncoding): An optional positional encoding
            object that may apply on each input.
        attention_module (Type[Module]): Sets the internal attention
            (self-attention and cross attention) class. By default, uses
            Multi-Head Attention.
        name (str): Default name is in the form
            'transformerdecoderlayer<index>'.

    """

    global_count = 0  # Static counter, used for default names

    def __init__(
        self,
        embed_dim,
        num_heads,
        feedforward_dim,
        dropout,
        attn_dropout,
        pre_layernorm=False,
        activation=lbann.Relu,
        parallel_attention_heads=0,
        attention_bias=None,
        positional_encoding: Optional[SequenceEncoding] = None,
        attention_module=lbann.modules.MultiheadAttention,
        name=None,
    ):
        TransformerDecoderLayer.global_count += 1
        self.instance = 0
        self.embed_dim = embed_dim
        self.feedforward_dim = feedforward_dim
        self.dropout_prob = dropout
        self.pre_layernorm = pre_layernorm
        self.activation = activation
        self.extra_ffn_args = {}
        self.extra_layer_args = {}

        # Module name
        self.name = name
        if not self.name:
            self.name = f'transformerdecoderlayer{TransformerDecoderLayer.global_count}'

        # Layer modules
        self.attention1 = attention_module(
            embed_dim,
            num_heads,
            self_attention=True,
            dropout=attn_dropout,
            subgraph_branches=parallel_attention_heads,
            bias=attention_bias,
            positional_encoding=positional_encoding,
            name=f'{self.name}_attention1')
        self.attention2 = attention_module(
            embed_dim,
            num_heads,
            dropout=attn_dropout,
            subgraph_branches=parallel_attention_heads,
            bias=attention_bias,
            name=f'{self.name}_attention2')
        self.norm1 = LayerNorm(self.embed_dim, name=f'{self.name}_norm1')
        self.norm2 = LayerNorm(self.embed_dim, name=f'{self.name}_norm2')
        self.norm3 = LayerNorm(self.embed_dim, name=f'{self.name}_norm3')

        # Weights for fully-connected layers
        self.fc1_weights = [
            lbann.Weights(initializer=lbann.HeNormalInitializer(),
                          name=f'{self.name}_fc1_matrix'),
            lbann.Weights(initializer=lbann.ConstantInitializer(value=0),
                          name=f'{self.name}_fc1_bias'),
        ]
        self.fc2_weights = [
            lbann.Weights(initializer=lbann.GlorotNormalInitializer(),
                          name=f'{self.name}_fc2_matrix'),
            lbann.Weights(initializer=lbann.ConstantInitializer(value=0),
                          name=f'{self.name}_fc2_bias'),
        ]

    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        """Apply Transformer decoder layer.

        Args:
            x (lbann.Layer): Sequence of input vectors.
            memory (lbann.Layer): Sequence of vectors produced by
                Transformer encoder stack, or None to disable cross-attention.
            src_mask (lbann.Layer, optional): Attention mask for
                second attention module (attends to both `x` and
                `memory`).
            tgt_mask (lbann.Layer, optional): Attention mask for first
                attention module (attends only to `x`).

        Returns:
            lbann.Layer: Sequence of output vectors.

        """
        self.instance += 1
        name = f'{self.name}_instance{self.instance}'

        if self.pre_layernorm:
            y = self.norm1(x, **self.extra_layer_args)
        else:
            y = x

        # Self-attention with residual connection
        y = self.attention1(y, y, y, mask=tgt_mask, **self.extra_layer_args)
        if self.dropout_prob > 0:
            y = lbann.Dropout(
                y,
                keep_prob=1 - self.dropout_prob,
                name=f'{name}_drop1',
                **self.extra_layer_args,
            )
        z = lbann.Sum(x, y, name=f'{name}_sum1', **self.extra_layer_args)

        if not self.pre_layernorm:
            z = self.norm1(z, **self.extra_layer_args)

        x = z

        # Cross-attention
        if memory is not None:
            # Attention on encoder output with residual connection
            if self.pre_layernorm:
                y = self.norm2(x, **self.extra_layer_args)
            else:
                y = x

            y = self.attention2(y,
                                memory,
                                memory,
                                mask=src_mask,
                                **self.extra_layer_args)
            if self.dropout_prob > 0:
                y = lbann.Dropout(y,
                                  keep_prob=1 - self.dropout_prob,
                                  name=f'{name}_drop2',
                                  **self.extra_layer_args)
            z = lbann.Sum(x, y, name=f'{name}_sum2', **self.extra_layer_args)

            if not self.pre_layernorm:
                z = self.norm2(z, **self.extra_layer_args)

            x = z

        # Feedforward network with residual connection
        if self.pre_layernorm:
            y = self.norm3(x, **self.extra_layer_args)
        else:
            y = x

        y = lbann.ChannelwiseFullyConnected(
            y,
            weights=self.fc1_weights,
            output_channel_dims=[self.feedforward_dim],
            name=f'{name}_fc1',
            **self.extra_layer_args,
            **self.extra_ffn_args,
        )
        y = self.activation(y,
                            name=f'{name}_ffn_act',
                            **self.extra_layer_args,
                            **self.extra_ffn_args)
        if self.dropout_prob > 0:
            y = lbann.Dropout(
                y,
                keep_prob=1 - self.dropout_prob,
                name=f'{name}_drop3',
                **self.extra_layer_args,
                **self.extra_ffn_args,
            )
        y = lbann.ChannelwiseFullyConnected(
            y,
            weights=self.fc2_weights,
            output_channel_dims=[self.embed_dim],
            name=f'{name}_fc2',
            **self.extra_layer_args,
            **self.extra_ffn_args,
        )
        if self.dropout_prob > 0:
            y = lbann.Dropout(
                y,
                keep_prob=1 - self.dropout_prob,
                name=f'{name}_drop4',
                **self.extra_layer_args,
                **self.extra_ffn_args,
            )
        z = lbann.Sum(x, y, name=f'{name}_sum3', **self.extra_layer_args)

        if not self.pre_layernorm:
            z = self.norm3(z, **self.extra_layer_args)

        return z


class Transformer(lbann.modules.Module):
    """Transformer model.

    See:

    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion
    Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin.
    "Attention is all you need." In Advances in Neural Information
    Processing Systems, pp. 5998-6008. 2017.

    Args:
        hidden_size (int): Internal dimensionality of multi-head
            attention.
        num_heads (int): Number of attention heads.
        num_encoder_layers (int): Number of stacked layers in encoder.
        num_decoder_layers (int): Number of stacked layers in decoder.
        feedforward_size (int): Internal dimensionality of fully-connected
            feedforward networks.
        dropout (float): Dropout probability after multi-head attention.
        attn_dropout (float): Dropout probability during multi-head attention.
        pre_layernorm (bool): If True, performs layer normalization before
            applying attention operators.
        activation (Type[lbann.Layer]): Activation function to apply in
            feedforward network. Examples include ReLU or GELU.
        parallel_attention_heads (int): If positive, applies subgraph
            parallelism on attention heads. 
        attention_bias (Layer): Additive attention bias to apply on the attention
            probability matrix before softmax. If None, does not apply.
        positional_encoding (SequenceEncoding): An optional positional encoding
            object that may apply on each input, in each layer.
        attention_module (Type[Module]): Sets the internal attention
            (self-attention and cross attention) class. By default, uses
            Multi-Head Attention.
        name (str): Default name is in the form
            'transformer<index>'.

    """

    global_count = 0  # Static counter, used for default names

    def __init__(
        self,
        hidden_size=512,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        feedforward_size=None,
        dropout=0.1,
        attn_dropout=0.0,
        pre_layernorm=False,
        activation=lbann.Relu,
        parallel_attention_heads=0,
        attention_bias=None,
        positional_encoding=None,
        attention_module=lbann.modules.MultiheadAttention,
        name=None,
    ):
        Transformer.global_count += 1
        self.instance = 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.pre_layernorm = pre_layernorm
        self.activation = activation
        self.parallel_attention_heads = parallel_attention_heads

        # Module name
        self.name = name
        if not self.name:
            self.name = f'transformer{Transformer.global_count}'

        # Caches for helper functions
        self._subsequent_mask_cache = {}

        # Default feedforward size is 4*hidden_size
        if feedforward_size is None or feedforward_size == 0:
            feedforward_size = 4 * hidden_size
        self.feedforward_size = feedforward_size

        # Encoder and decoder stacks
        self.encoder = [
            TransformerEncoderLayer(
                embed_dim=hidden_size,
                num_heads=num_heads,
                feedforward_dim=feedforward_size,
                dropout=dropout,
                attn_dropout=attn_dropout,
                pre_layernorm=pre_layernorm,
                activation=activation,
                parallel_attention_heads=parallel_attention_heads,
                attention_bias=attention_bias,
                positional_encoding=positional_encoding,
                attention_module=attention_module,
                name=f'{self.name}_encoder{i}',
            ) for i in range(num_encoder_layers)
        ]
        self.decoder = [
            TransformerDecoderLayer(
                embed_dim=hidden_size,
                num_heads=num_heads,
                feedforward_dim=feedforward_size,
                dropout=dropout,
                attn_dropout=attn_dropout,
                pre_layernorm=pre_layernorm,
                activation=activation,
                parallel_attention_heads=parallel_attention_heads,
                attention_bias=attention_bias,
                positional_encoding=positional_encoding,
                attention_module=attention_module,
                name=f'{self.name}_decoder{i}',
            ) for i in range(num_decoder_layers)
        ]
        self.separate_heads = self.decoder[0].attention1.separate_heads
        assert all(dec.attention1.separate_heads == self.separate_heads
                   for dec in self.decoder)

    def _subsequent_mask(self, size):
        """Attention mask to prevent attending to subsequent positions.

        The (i,j) entry is -1e9 if i<j and is 0 otherwise. Masks are
        memoized.

        """

        # Construct mask if not in cache
        if size not in self._subsequent_mask_cache:
            vals = np.triu(np.full((size, size), -1e9), k=1)

            if not self.separate_heads:
                # Precompute mask for all heads because Add is entry-wise
                # (potential memory usage issue)
                vals = np.tile(vals, (self.num_heads, 1, 1))

            weights = lbann.Weights(
                initializer=lbann.ValueInitializer(values=vals.flat),
                optimizer=None,
                name=f'{self.name}_mask{size}_weights',
            )
            self._subsequent_mask_cache[size] = lbann.WeightsLayer(
                dims=vals.shape,
                weights=weights,
                name=f'{self.name}_mask{size}',
            )

        # Return cached mask
        return self._subsequent_mask_cache[size]

    def forward(self,
                source: lbann.Layer,
                target: lbann.Layer,
                target_length: int,
                target_mask: Optional[lbann.Layer] = None,
                cross_attention: Optional[lbann.Layer] = None):
        """Apply Transformer.

        The input and output tensors are interpreted as sequences of
        vectors, where the first tensor dimension is the sequence
        dimension.

        Args:
            source: Sequence of input vectors to encoder stack.
            target: Sequence of input vectors to decoder stack.
            target_length: Length of input sequence to decoder.
            target_mask: Optional mask tensor for different decoder masking
                         schemes.
            cross_attention: Optional cross-attention tensor to give to a
                decoder-only architecture.

        Returns:
            lbann.Layer: Sequence of output vectors.
        """
        self.instance += 1

        # Encoder stack (assumes encoded input, including positional encoding)
        if self.encoder:
            x = source
            for encoder_layer in self.encoder:
                x = encoder_layer(x)
            memory = x
        else:
            # Decoder-only architecture with an optional input cross-attention
            # tensor
            memory = cross_attention

        if not self.decoder:  # Encoder-only architecture
            return x

        # Decoder stack
        x = target

        # Create mask if not given
        if target_mask is None:
            target_mask = self._subsequent_mask(target_length)

        # For attention-head parallelism, replicate mask for each subgraph
        if self.parallel_attention_heads > 0:
            target_mask = [
                lbann.Identity(target_mask,
                               name=f'tgtmask_branch{i}',
                               parallel_strategy=dict(grid_tag=i))
                for i in range(1, self.parallel_attention_heads + 1)
            ]

        for decoder_layer in self.decoder:
            x = decoder_layer(x, memory, tgt_mask=target_mask)

        return x
