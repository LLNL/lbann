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

    """

    global_count = 0  # Static counter, used for default names

    def __init__(
        self,
        embed_dim=512,
        num_heads=8,
        feedforward_dim=2048,
        name=None,
    ):
        TransformerEncoderLayer.global_count += 1
        self.instance = 0
        self.name = (name
                     if name
                     else f'transformerencoderlayer{TransformerEncoderLayer.global_count}')

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
        self.instance += 1
        sequence_length = len(x)

        # Self-attention with residual connection
        y = self.attention(x, x, x, mask=mask)
        z = []
        for i in range(sequence_length):
            xi = x[i]
            yi = y[i]
            zi = lbann.LayerNorm(
                lbann.Sum(
                    xi, yi,
                    name=f'{self.name}_sum1_pos{i}_instance{self.instance}'
                ),
                name=f'{self.name}_norm1_pos{i}_instance{self.instance}',
            )
            z.append(zi)
        x = z

        # Feedforward network with residual connection
        z = []
        for i in range(sequence_length):
            xi = x[i]
            yi = self.fc2(self.fc1(xi))
            zi = lbann.LayerNorm(
                lbann.Sum(
                    xi, yi,
                    name=f'{self.name}_sum2_pos{i}_instance{self.instance}'
                ),
                name=f'{self.name}_norm2_pos{i}_instance{self.instance}',
            )
            z.append(zi)

        return z

class TransformerDecoderLayer(lbann.modules.Module):
    """Building block for decoder in transformer model.

    Comprised of two multi-head attention modules and a
    fully-connected feedforward network, each with a residual
    connection.

    """

    global_count = 0  # Static counter, used for default names

    def __init__(
        self,
        embed_dim=512,
        num_heads=8,
        feedforward_dim=2048,
        name=None,
    ):
        TransformerDecoderLayer.global_count += 1
        self.instance = 0
        self.name = (name
                     if name
                     else f'transformerdecoderlayer{TransformerEncoderLayer.global_count}')

        # Layer modules
        # TODO: Masked attention
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
        self.instance += 1
        sequence_length = len(x)

        # Self-attention with residual connection
        y = self.attention1(x, x, x, mask=tgt_mask)
        z = []
        for i in range(sequence_length):
            xi = x[i]
            yi = y[i]
            zi = lbann.LayerNorm(
                lbann.Sum(
                    xi, yi,
                    name=f'{self.name}_sum1_pos{i}_instance{self.instance}'
                ),
                name=f'{self.name}_norm1_pos{i}_instance{self.instance}',
            )
            z.append(zi)
        x = z

        # Attention on encoder output with residual connection
        y = self.attention2(x, memory, memory, mask=src_mask)
        z = []
        for i in range(sequence_length):
            xi = x[i]
            yi = y[i]
            zi = lbann.LayerNorm(
                lbann.Sum(
                    xi, yi,
                    name=f'{self.name}_sum2_pos{i}_instance{self.instance}'
                ),
                name=f'{self.name}_norm2_pos{i}_instance{self.instance}',
            )
            z.append(zi)
        x = z

        # Feedforward network with residual connection
        z = []
        for i in range(sequence_length):
            xi = x[i]
            yi = self.fc2(self.fc1(xi))
            zi = lbann.LayerNorm(
                lbann.Sum(
                    xi, yi,
                    name=f'{self.name}_sum3_pos{i}_instance{self.instance}'
                ),
                name=f'{self.name}_norm3_pos{i}_instance{self.instance}',
            )
            z.append(zi)

        return z

class Transformer(lbann.modules.Module):
    """Basic transformer model.

    See:

    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion
    Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin.
    "Attention is all you need." In Advances in Neural Information
    Processing Systems, pp. 5998-6008. 2017.

    """

    global_count = 0  # Static counter, used for default names

    def __init__(
        self,
        hidden_size=512,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        filter_size=2048,
        name=None,
    ):
        Transformer.global_count += 1
        self.instance = 0
        self.name = (name
                     if name
                     else f'transformer{Transformer.global_count}')
        self.hidden_size = hidden_size

        # Caches for helper functions
        self.subsequent_mask_cache = {}
        self.positional_encoding_cache = {}

        # Encoder and decoder stacks
        self.encoder = [
            TransformerEncoderLayer(
                embed_dim=hidden_size,
                num_heads=num_heads,
                feedforward_dim=filter_size,
                name=f'{self.name}_encoder{i}',
            )
            for i in range(num_encoder_layers)
        ]
        self.decoder = [
            TransformerDecoderLayer(
                embed_dim=hidden_size,
                num_heads=num_heads,
                feedforward_dim=filter_size,
                name=f'{self.name}_decoder{i}',
            )
            for i in range(num_decoder_layers)
        ]

    def positional_encoding(self, pos):

        # Construct positional encoding if not in cache
        if pos not in self.positional_encoding_cache:
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
            self.positional_encoding_cache[pos] = lbann.WeightsLayer(
                dims=str(self.hidden_size),
                weights=weights,
                name=f'{self.name}_positional{pos}',
            )

        # Return cached positional encoding
        return self.positional_encoding_cache[pos]

    def subsequent_mask(self, size):

        # Construct mask if not in cache
        if size not in self.subsequent_mask_cache:
            vals = np.triu(np.full((size,size), -1e9), k=1)
            weights = lbann.Weights(
                initializer=lbann.ValueInitializer(values=str_list(np.nditer(vals))),
                optimizer=None,
                name=f'{self.name}_mask{size}_weights',
            )
            self.subsequent_mask_cache[size] = lbann.WeightsLayer(
                dims=str_list([size, size]),
                weights=weights,
                name=f'{self.name}_mask{size}',
            )

        # Return cached mask
        return self.subsequent_mask_cache[size]

    def forward(self, source, target):
        self.instance += 1

        # Add positional encoding
        source = source.copy()
        target = target.copy()
        for pos in range(len(source)):
            source[pos] = lbann.Add(
                source[pos],
                self.positional_encoding(pos),
                name=f'{self.name}_instance{self.instance}_positional_source{pos}',
            )
        for pos in range(len(target)):
            target[pos] = lbann.Add(
                target[pos],
                self.positional_encoding(pos),
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
                tgt_mask=self.subsequent_mask(len(target)),
            )

        return x
