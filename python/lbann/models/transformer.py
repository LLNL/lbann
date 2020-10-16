"""Basic Transformer model with multi-head self-attention.

See:

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion
Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. "Attention
is all you need." In Advances in Neural Information Processing
Systems, pp. 5998-6008. 2017.

"""
import math
import numpy as np

import lbann
import lbann.modules
from lbann.util import str_list

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
        dropout (float): Dropout probability.
        name (str): Default name is in the form
            'transformerencoderlayer<index>'.

    """

    global_count = 0  # Static counter, used for default names

    def __init__(
        self,
        branches,
        embed_dim=512,
        num_heads=8,
        feedforward_dim=2048,
        dropout=0.1,
        d_kv=None,
        name=None,
    ):
        TransformerEncoderLayer.global_count += 1
        self.instance = 0
        self.embed_dim = embed_dim
        self.feedforward_dim = feedforward_dim
        self.dropout_prob = dropout

        # Module name
        self.name = name
        if not self.name:
            self.name = f'transformerencoderlayer{TransformerEncoderLayer.global_count}'

        # Layer modules
        self.attention = lbann.modules.transformer.MultiheadAttention(
            self.embed_dim,
            num_heads,
            branches=branches,
            d_kv = d_kv,
            name=f'{self.name}_attention'
        )

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

        # Self-attention with residual connection
        y = self.attention(x, x, x, mask=mask)
        if self.dropout_prob > 0:
            y = lbann.Dropout(
                y,
                keep_prob=1-self.dropout_prob,
                name=f'{name}_drop1',
            )
        z = lbann.Sum(x, y, name=f'{name}_sum1')
        z = lbann.InstanceNorm(z, name=f'{name}_norm1')
        x = z

        # Feedforward network with residual connection
        y = lbann.ChannelwiseFullyConnected(
            x,
            weights=self.fc1_weights,
            output_channel_dims=[self.feedforward_dim],
            name=f'{name}_fc1',
        )
        y = lbann.Relu(y, name=f'{name}_relu1')
        if self.dropout_prob > 0:
            y = lbann.Dropout(
                y,
                keep_prob=1-self.dropout_prob,
                name=f'{name}_drop2',
            )
        y = lbann.ChannelwiseFullyConnected(
            y,
            weights=self.fc2_weights,
            output_channel_dims=[self.embed_dim],
            name=f'{name}_fc2',
        )
        if self.dropout_prob > 0:
            y = lbann.Dropout(
                y,
                keep_prob=1-self.dropout_prob,
                name=f'{name}_drop3',
            )
        z = lbann.Sum(x, y, name=f'{name}_sum2')
        z = lbann.InstanceNorm(z, name=f'{name}_norm2')
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
        dropout (float): Dropout probability.
        name (str): Default name is in the form
            'transformerdecoderlayer<index>'.

    """

    global_count = 0  # Static counter, used for default names

    def __init__(
        self,
        branches,
        embed_dim=512,
        num_heads=8,
        feedforward_dim=2048,
        dropout=0.1,
        d_kv = None,
        name=None,
    ):
        TransformerDecoderLayer.global_count += 1
        self.instance = 0
        self.embed_dim = embed_dim
        self.feedforward_dim = feedforward_dim
        self.dropout_prob = dropout

        # Module name
        self.name = name
        if not self.name:
            self.name = f'transformerdecoderlayer{TransformerDecoderLayer.global_count}'

        # Layer modules
        self.attention1 = lbann.modules.transformer.MultiheadAttention(
            embed_dim,
            num_heads,
            branches=branches,
            d_kv = d_kv,
            name=f'{self.name}_attention1'
        )
        self.attention2 = lbann.modules.transformer.MultiheadAttention(
            embed_dim,
            num_heads,
            branches=branches,
            d_kv = d_kv,
            name=f'{self.name}_attention2'
        )

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
                Transformer encoder stack.
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

        # Self-attention with residual connection
        y = self.attention1(x, x, x, mask=tgt_mask)
        if self.dropout_prob > 0:
            y = lbann.Dropout(
                y,
                keep_prob=1-self.dropout_prob,
                name=f'{name}_drop1',
            )
        z = lbann.Sum(x, y, name=f'{name}_sum1')
        z = lbann.InstanceNorm(z, name=f'{name}_norm1')
        x = z

        # Attention on encoder output with residual connection
        y = self.attention2(x, memory, memory, mask=src_mask)
        if self.dropout_prob > 0:
            y = lbann.Dropout(
                y,
                keep_prob=1-self.dropout_prob,
                name=f'{name}_drop2',
            )
        z = lbann.Sum(x, y, name=f'{name}_sum2')
        z = lbann.InstanceNorm(z, name=f'{name}_norm2')
        x = z

        # Feedforward network with residual connection
        y = lbann.ChannelwiseFullyConnected(
            x,
            weights=self.fc1_weights,
            output_channel_dims=[self.feedforward_dim],
            name=f'{name}_fc1',
        )
        y = lbann.Relu(y, name=f'{name}_relu1')
        if self.dropout_prob > 0:
            y = lbann.Dropout(
                y,
                keep_prob=1-self.dropout_prob,
                name=f'{name}_drop3',
            )
        y = lbann.ChannelwiseFullyConnected(
            y,
            weights=self.fc2_weights,
            output_channel_dims=[self.embed_dim],
            name=f'{name}_fc2',
        )
        if self.dropout_prob > 0:
            y = lbann.Dropout(
                y,
                keep_prob=1-self.dropout_prob,
                name=f'{name}_drop4',
            )
        z = lbann.Sum(x, y, name=f'{name}_sum3')
        z = lbann.InstanceNorm(z, name=f'{name}_norm3')
        return z

class Transformer(lbann.modules.Module):
    """Transformer model.

    See:

    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion
    Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin.
    "Attention is all you need." In Advances in Neural Information
    Processing Systems, pp. 5998-6008. 2017.

    Args:
        hidden_dim (int): Internal dimensionality of multi-head
            attention.
        num_heads (int): Number of attention heads.
        num_encoder_layers (int): Number of stacked layers in encoder.
        num_decoder_layers (int): Number of stacked layers in decoder.
        filter_dim (int): Internal dimensionality of fully-connected
            feedforward networks.
        dropout (float): Dropout probability.
        name (str): Default name is in the form
            'transformer<index>'.

    """

    global_count = 0  # Static counter, used for default names

    def __init__(
        self,
        branches,
        hidden_size=512,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        filter_size=2048,
        dropout=0.1,
        d_kv = None,
        name=None,
    ):
        Transformer.global_count += 1
        self.instance = 0
        self.hidden_size = hidden_size

        # Module name
        self.name = name
        if not self.name:
            self.name = f'transformer{Transformer.global_count}'

        # Caches for helper functions
        self._subsequent_mask_cache = {}
        self._positional_encoding_cache = {}

        # Encoder and decoder stacks
        self.encoder = [
            TransformerEncoderLayer(
                branches,
                embed_dim=hidden_size,
                num_heads=num_heads,
                feedforward_dim=filter_size,
                dropout=dropout,
                d_kv = d_kv,
                name=f'{self.name}_encoder{i}',
            )
            for i in range(num_encoder_layers)
        ]
        self.decoder = [
            TransformerDecoderLayer(
                branches,
                embed_dim=hidden_size,
                num_heads=num_heads,
                feedforward_dim=filter_size,
                dropout=dropout,
                d_kv = d_kv,
                name=f'{self.name}_decoder{i}',
            )
            for i in range(num_decoder_layers)
        ]
        self.branches = branches

    def _positional_encoding(self, sequence_length):
        """Positional encodings corresponding to a sequence length.

        PE(pos,2*i)   = sin( pos / 10000**(2*i/hidden_size) )

        PE(pos,2*i+1) = cos( pos / 10000**(2*i/hidden_size) )

        Encodings are memoized.

        """

        # Construct positional encoding if not in cache
        if sequence_length not in self._positional_encoding_cache:
            vals = []
            for pos in range(sequence_length):
                for i in range((self.hidden_size+1) // 2):
                    x = pos / 10000**(2*i/self.hidden_size)
                    vals.append(math.sin(x))
                    vals.append(math.cos(x))
                if self.hidden_size % 2 != 0:
                    vals.pop()
            weights = lbann.Weights(
                initializer=lbann.ValueInitializer(values=str_list(vals)),
                optimizer=None,
                name=f'{self.name}_positional{sequence_length}_weights',
            )
            self._positional_encoding_cache[sequence_length] = lbann.WeightsLayer(
                dims=str_list([sequence_length, self.hidden_size]),
                weights=weights,
                name=f'{self.name}_positional{sequence_length}',
            )

        # Return cached positional encoding
        return self._positional_encoding_cache[sequence_length]

    def _subsequent_mask(self, size):
        """Attention mask to prevent attending to subsequent positions.

        The (i,j) entry is -1e9 if i<j and is 0 otherwise. Masks are
        memoized.

        """

        # Construct mask if not in cache
        if size not in self._subsequent_mask_cache:
            vals = np.triu(np.full((size,size), -1e9), k=1)
            weights = lbann.Weights(
                initializer=lbann.ValueInitializer(values=str_list(np.nditer(vals))),
                optimizer=None,
                name=f'{self.name}_mask{size}_weights',
            )
            self._subsequent_mask_cache[size] = lbann.WeightsLayer(
                dims=str_list([size, size]),
                weights=weights,
                name=f'{self.name}_mask{size}',
            )

        # Return cached mask
        return self._subsequent_mask_cache[size]

    def forward(self, source, source_length, target, target_length):
        """Apply Transformer.

        The input and output tensors are interpreted as sequences of
        vectors, where the first tensor dimension is the sequence
        dimension.

        Args:
            source (lbann.Layer): Sequence of input vectors to encoder
                stack.
            source_length (int): Length of input sequence to encoder.
            target (lbann.Layer): Sequence of input vectors to decoder
                stack.
            target_length (int): Length of input sequence to decoder.

        Returns:
            lbann.Layer: Sequence of output vectors.

        """
        self.instance += 1

        # Encoder stack
        # Note: Add positional encoding to input
        x = lbann.Sum([
            source,
            self._positional_encoding(source_length)],
            name=f'{self.name}_instance{self.instance}_positional_source',
        )
        for encoder_layer in self.encoder:
            x = encoder_layer(x)
        memory = x

        # Decoder stack
        # Note: Add positional encoding to input
        x = lbann.Sum(
            [target,
            self._positional_encoding(target_length)],
            name=f'{self.name}_instance{self.instance}_positional_target',
        )

        subgraph_masks = {}


        if(self.branches>0):
            for i in range(self.branches):
                subgraph_masks[i+1] = lbann.Identity(self._subsequent_mask(target_length),name="mylayer"+str(i) ,
                                    parallel_strategy = {'sub_branch_tag':i+1,'enable_subgraph':1})
                subgraph_masks[i+1] = lbann.Identity(subgraph_masks[i+1])
            

        if(self.branches>0):
            for decoder_layer in self.decoder:
                x = decoder_layer(
                    x,
                    memory,
                    tgt_mask=subgraph_masks,
                )

        else:
            for decoder_layer in self.decoder:
                x = decoder_layer(
                    x,
                    memory,
                    tgt_mask=self._subsequent_mask(target_length),
                )

        return x
