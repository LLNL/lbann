"""Basic transformer model with multi-head self-attention.

See:

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion
Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. "Attention
is all you need." In Advances in Neural Information Processing
Systems, pp. 5998-6008. 2017.

"""
import lbann
import lbann.modules

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

    def forward(self, x):
        self.instance += 1
        sequence_length = len(x)

        # Self-attention with residual connection
        y = self.attention(x, x, x)
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

    def forward(self, x, memory):
        self.instance += 1
        sequence_length = len(x)

        # Self-attention with residual connection
        y = self.attention1(x, x, x)
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
        y = self.attention2(x, memory, memory)
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

    def forward(self, source, target):
        self.instance += 1

        # Encoder stack
        # TODO: Positional encoding
        x = source
        for encoder_layer in self.encoder:
            x = encoder_layer(x)
        memory = x

        # Decoder stack
        # TODO: Positional encoding
        x = target
        for decoder_layer in self.decoder:
            x = decoder_layer(x, memory)

        return x
