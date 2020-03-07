"""Basic transformer model with multi-head self-attention.

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
    """Building block for encoder in transformer model.

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
        embed_dim=512,
        num_heads=8,
        feedforward_dim=2048,
        dropout=0.1,
        name=None,
    ):
        TransformerEncoderLayer.global_count += 1
        self.instance = 0
        self.name = (name
                     if name
                     else f'transformerencoderlayer{TransformerEncoderLayer.global_count}')
        self.dropout_prob = dropout

        # Layer modules
        self.attention = lbann.modules.transformer.MultiheadAttention(
            embed_dim,
            num_heads,
            name=f'{self.name}_attention'
        )
        self.fc1 = lbann.modules.FullyConnectedModule(
            feedforward_dim,
            activation=lbann.Relu,
            name=f'{self.name}_fc1',
        )
        self.fc2 = lbann.modules.FullyConnectedModule(
            embed_dim,
            name=f'{self.name}_fc2',
        )

    def forward(self, x, mask=None):
        """Apply transformer encoder layer.

        Args:
            x (Iterable of lbann.Layer): Sequence of input vectors.
            mask (lbann.Layer, optional): Attention mask.

        Returns:
            list of lbann.Layer: Sequence of output vectors.

        """
        self.instance += 1
        sequence_length = len(x)

        # Self-attention with residual connection
        y = self.attention(x, x, x, mask=mask)
        z = []
        for i in range(sequence_length):
            name = f'{self.name}_instance{self.instance}_pos{i}'
            xi = x[i]
            yi = y[i]
            if self.dropout_prob > 0:
                yi = lbann.Dropout(
                    yi,
                    keep_prob=1-self.dropout_prob,
                    name=f'{name}_drop1',
                )
            zi = lbann.Sum(xi, yi, name=f'{name}_sum1')
            zi = lbann.LayerNorm(zi, name=f'{name}_norm1')
            z.append(zi)
        x = z

        # Feedforward network with residual connection
        z = []
        for i in range(sequence_length):
            name = f'{self.name}_instance{self.instance}_pos{i}'
            xi = x[i]
            yi = self.fc2(self.fc1(xi))
            if self.dropout_prob > 0:
                yi = lbann.Dropout(
                    yi,
                    keep_prob=1-self.dropout_prob,
                    name=f'{name}_drop2',
                )
            zi = lbann.Sum(xi, yi, name=f'{name}_sum2')
            zi = lbann.LayerNorm(zi, name=f'{name}_norm2')
            z.append(zi)

        return z

class TransformerDecoderLayer(lbann.modules.Module):
    """Building block for decoder in transformer model.

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
        embed_dim=512,
        num_heads=8,
        feedforward_dim=2048,
        dropout=0.1,
        name=None,
    ):
        TransformerDecoderLayer.global_count += 1
        self.instance = 0
        self.name = (name
                     if name
                     else f'transformerdecoderlayer{TransformerEncoderLayer.global_count}')
        self.dropout_prob = dropout

        # Layer modules
        self.attention1 = lbann.modules.transformer.MultiheadAttention(
            embed_dim,
            num_heads,
            name=f'{self.name}_attention1'
        )
        self.attention2 = lbann.modules.transformer.MultiheadAttention(
            embed_dim,
            num_heads,
            name=f'{self.name}_attention2'
        )
        self.fc1 = lbann.modules.FullyConnectedModule(
            feedforward_dim,
            activation=lbann.Relu,
            name=f'{self.name}_fc1',
        )
        self.fc2 = lbann.modules.FullyConnectedModule(
            embed_dim,
            name=f'{self.name}_fc2',
        )

    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        """Apply transformer decoder layer.

        Args:
            x (Iterable of lbann.Layer): Sequence of input vectors.
            memory (Iterable of lbann.Layer): Sequence of vectors
                produced by transformer encoder stack.
            src_mask (lbann.Layer, optional): Attention mask for
                second attention module (attends to both `x` and
                `memory`).
            tgt_mask (lbann.Layer, optional): Attention mask for first
                attention module (attends only to `x`).

        Returns:
            list of lbann.Layer: Sequence of output vectors.

        """
        self.instance += 1
        sequence_length = len(x)

        # Self-attention with residual connection
        y = self.attention1(x, x, x, mask=tgt_mask)
        z = []
        for i in range(sequence_length):
            name = f'{self.name}_instance{self.instance}_pos{i}'
            xi = x[i]
            yi = y[i]
            if self.dropout_prob > 0:
                yi = lbann.Dropout(
                    yi,
                    keep_prob=1-self.dropout_prob,
                    name=f'{name}_drop1',
                )
            zi = lbann.Sum(xi, yi, name=f'{name}_sum1')
            zi = lbann.LayerNorm(zi, name=f'{name}_norm1')
            z.append(zi)
        x = z

        # Attention on encoder output with residual connection
        y = self.attention2(x, memory, memory, mask=src_mask)
        z = []
        for i in range(sequence_length):
            name = f'{self.name}_instance{self.instance}_pos{i}'
            xi = x[i]
            yi = y[i]
            if self.dropout_prob > 0:
                yi = lbann.Dropout(
                    yi,
                    keep_prob=1-self.dropout_prob,
                    name=f'{name}_drop2',
                )
            zi = lbann.Sum(xi, yi, name=f'{name}_sum2')
            zi = lbann.LayerNorm(zi, name=f'{name}_norm2')
            z.append(zi)
        x = z

        # Feedforward network with residual connection
        z = []
        for i in range(sequence_length):
            name = f'{self.name}_instance{self.instance}_pos{i}'
            xi = x[i]
            yi = self.fc2(self.fc1(xi))
            if self.dropout_prob > 0:
                yi = lbann.Dropout(
                    yi,
                    keep_prob=1-self.dropout_prob,
                    name=f'{name}_drop3',
                )
            zi = lbann.Sum(xi, yi, name=f'{name}_sum3')
            zi = lbann.LayerNorm(zi, name=f'{name}_norm3')
            z.append(zi)

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
        hidden_size=512,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        filter_size=2048,
        dropout=0.1,
        name=None,
    ):
        Transformer.global_count += 1
        self.instance = 0
        self.name = (name
                     if name
                     else f'transformer{Transformer.global_count}')
        self.hidden_size = hidden_size

        # Caches for helper functions
        self._subsequent_mask_cache = {}
        self._positional_encoding_cache = {}

        # Encoder and decoder stacks
        self.encoder = [
            TransformerEncoderLayer(
                embed_dim=hidden_size,
                num_heads=num_heads,
                feedforward_dim=filter_size,
                dropout=dropout,
                name=f'{self.name}_encoder{i}',
            )
            for i in range(num_encoder_layers)
        ]
        self.decoder = [
            TransformerDecoderLayer(
                embed_dim=hidden_size,
                num_heads=num_heads,
                feedforward_dim=filter_size,
                dropout=dropout,
                name=f'{self.name}_decoder{i}',
            )
            for i in range(num_decoder_layers)
        ]

    def _positional_encoding(self, pos):
        """Positional encoding corresponding to a sequence position.

        PE(pos,2*i)   = sin( pos / 10000**(2*i/hidden_size) )

        PE(pos,2*i+1) = cos( pos / 10000**(2*i/hidden_size) )

        Encodings are memoized.

        """

        # Construct positional encoding if not in cache
        if pos not in self._positional_encoding_cache:
            vals = []
            for i in range((self.hidden_size+1) // 2):
                x = pos / 10000**(2*i/self.hidden_size)
                vals.append(math.sin(x))
                vals.append(math.cos(x))
            weights = lbann.Weights(
                initializer=lbann.ValueInitializer(values=str_list(vals[:self.hidden_size])),
                optimizer=None,
                name=f'{self.name}_positional{pos}_weights',
            )
            self._positional_encoding_cache[pos] = lbann.WeightsLayer(
                dims=str(self.hidden_size),
                weights=weights,
                name=f'{self.name}_positional{pos}',
            )

        # Return cached positional encoding
        return self._positional_encoding_cache[pos]

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

    def forward(self, source, target):
        """Apply transformer.

        Args:
            source (Iterable of lbann.Layer): Sequence of input
                vectors to encoder stack.
            target (Iterable of lbann.Layer): Sequence of input
                vectors to decoder stack.

        Returns:
            list of lbann.Layer: Sequence of output vectors.

        """
        self.instance += 1

        # Add positional encoding
        source = source.copy()
        target = target.copy()
        for pos in range(len(source)):
            source[pos] = lbann.Add(
                source[pos],
                self._positional_encoding(pos),
                name=f'{self.name}_instance{self.instance}_positional_source{pos}',
            )
        for pos in range(len(target)):
            target[pos] = lbann.Add(
                target[pos],
                self._positional_encoding(pos),
                name=f'{self.name}_instance{self.instance}_positional_target{pos}',
            )

        # Encoder stack
        x = source
        for encoder_layer in self.encoder:
            x = encoder_layer(x)
        memory = x

        # Decoder stack
        x = target
        for decoder_layer in self.decoder:
            x = decoder_layer(
                x,
                memory,
                tgt_mask=self._subsequent_mask(len(target)),
            )

        return x
