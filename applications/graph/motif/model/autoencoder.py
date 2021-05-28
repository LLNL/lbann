import lbann
import lbann.modules

class FullyConnectedAutoencoder(lbann.modules.Module):
    """Multilayer perceptron autoencoder."""

    global_count = 0  # Static counter, used for default names

    def __init__(
            self,
            data_dim,
            latent_dim,
            encoder_hidden_dims=[],
            decoder_hidden_dims=[],
            activation=lbann.Relu,
            data_layout='data_parallel',
            name=None,
    ):
        super().__init__()
        FullyConnectedAutoencoder.global_count += 1

        # Module name
        self.name = name
        if not self.name:
            self.name = f'fcautoencoder{FullyConnectedAutoencoder.global_count}'

        # Encoder
        self.encoder = []
        for i, dim in enumerate(encoder_hidden_dims):
            self.encoder.append(
                lbann.modules.FullyConnectedModule(
                    size=dim,
                    bias=False,
                    activation=activation,
                    name=f'{self.name}_encoder{i}',
                    data_layout=data_layout,
                )
            )
        self.encoder.append(
            lbann.modules.FullyConnectedModule(
                size=latent_dim,
                bias=False,
                activation=activation,
                name=f'{self.name}_encoder{len(self.encoder)}',
                data_layout=data_layout,
            )
        )

        # Decoder
        self.decoder = []
        for i, dim in enumerate(decoder_hidden_dims):
            self.decoder.append(
                lbann.modules.FullyConnectedModule(
                    size=dim,
                    bias=False,
                    activation=activation,
                    name=f'{self.name}_decoder{i}',
                    data_layout=data_layout,
                )
            )
        self.decoder.append(
            lbann.modules.FullyConnectedModule(
                size=data_dim,
                bias=False,
                activation=activation,
                name=f'{self.name}_decoder{len(self.decoder)}',
                data_layout=data_layout,
            )
        )

    def forward(self, x):
        for l in self.encoder:
            x = l(x)
        for l in self.decoder:
            x = l(x)
        return x
