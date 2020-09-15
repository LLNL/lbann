import math
import lbann
import lbann.modules
from lbann.util import make_iterable

def str_list(l):
    """Convert an iterable object to a space-separated string."""
    return ' '.join(str(i) for i in make_iterable(l))

class GRUModule(lbann.modules.Module):

    global_count = 0  # Static counter, used for default names

    def __init__(
        self,
        hidden_size,
        num_layers=1,
        weights=[],
        name=None,
        device=None,
        datatype=None,
        weights_datatype=None,
    ):
        GRUModule.global_count += 1
        self.instance = 0
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.name = name if name else f'gru{GRUModule.global_count}'
        self.device = device
        self.datatype = datatype

        # Construct weights if needed
        self.weights = weights
        if not self.weights:
            scale = 1 / math.sqrt(self.hidden_size)
            init = lbann.UniformInitializer(min=-scale,max=scale)
            if weights_datatype is None:
                weights_datatype = self.datatype
            self.weights = []
            for i in range(self.num_layers):
                self.weights.extend(
                    lbann.Weights(
                        initializer=init,
                        name=f'{self.name}_layer{i}_{weight_name}',
                        datatype=weights_datatype,
                    )
                    for weight_name in ('ih_matrix', 'hh_matrix', 'ih_bias', 'hh_bias')
                )
        if self.weights and len(self.weights) != 4*self.num_layers:
            raise ValueError(
                f'expected {4*self.num_layers} weights, '
                f'but recieved {len(self.weights)}'
            )

        # Default initial hidden state
        self.zeros = lbann.Constant(
            value=0,
            num_neurons=str(hidden_size),
            name=f'{self.name}_zeros',
            device=self.device,
            datatype=self.datatype,
        )

    def forward(self, x, h=None):
        self.instance += 1
        name = f'{self.name}_instance{self.instance}'

        # Initial hidden state
        if not h:
            h = [self.zeros] * self.num_layers
        if not isinstance(h, list) or len(h) != self.num_layers:
            raise ValueError(
                f'expected `h` to be a list with {self.num_layers} layers'
            )

        # Stacked GRU
        ### @todo Replace with single GRU once LBANN supports stacked GRUs
        for i in range(self.num_layers):
            x = lbann.GRU(
                x,
                h[i],
                hidden_size=self.hidden_size,
                name=f'{name}_layer{i}',
                weights=self.weights[4*i:4*(i+1)],
                device=self.device,
                datatype=self.datatype,
            )
        return x

class MolVAE(lbann.modules.Module):
    """Molecular VAE.

    See:
    https://github.com/samadejacobs/moses/tree/master/moses/vae

    """

    global_count = 0  # Static counter, used for default names

    def __init__(self, input_feature_dims,dictionary_size, embedding_size, ignore_label, name=None):
        """Initialize Molecular VAE.

        Args:
            input_feature_dims (int): analogous to sequence length.
            dictionary_size (int): vocabulary size
            embedding_size (int): embedding size
            ignore_label (int): padding index
            name (str, optional): Module name
                (default: 'molvae_module<index>').

        """
        MolVAE.global_count += 1
        self.instance = 0
        self.name = (name if name
                     else 'molvae_module{0}'.format(MolVAE.global_count))

        self.input_feature_dims = input_feature_dims
        self.embedding_size = embedding_size
        self.dictionary_size = dictionary_size
        self.label_to_ignore = ignore_label
        self.datatype = lbann.DataType.FP16
        self.weights_datatype = lbann.DataType.FLOAT

        fc = lbann.modules.FullyConnectedModule
        gru = GRUModule

        #Encoder
        self.encoder_rnn = gru(
            hidden_size=256,
            name=self.name+'_encoder_rnn',
            datatype=self.datatype,
            weights_datatype=self.weights_datatype,
        )
        self.q_mu = fc(128,name=self.name+'_encoder_qmu')
        self.q_logvar = fc(128,name=self.name+'_encoder_qlogvar')
        for w in self.q_mu.weights + self.q_logvar.weights:
            w.datatype = self.weights_datatype

        #Decoder
        self.decoder_rnn = gru(
            hidden_size=512,
            num_layers=3,
            name=self.name+'_decoder_rnn',
            datatype=self.datatype,
            weights_datatype=self.weights_datatype,
        )
        self.decoder_lat = fc(512, name=self.name+'_decoder_lat')
        self.decoder_fc = fc(self.dictionary_size, name=self.name+'_decoder_fc')
        for w in self.decoder_lat.weights + self.decoder_fc.weights:
            w.datatype = self.weights_datatype
        self.decoder_fc.weights[0].initializer = lbann.NormalInitializer(
            mean=0, standard_deviation=1/math.sqrt(512))

        #shared encoder/decoder weights
        self.emb_weights = lbann.Weights(
            initializer=lbann.NormalInitializer(mean=0, standard_deviation=1),
            name='emb_matrix',
            datatype=self.weights_datatype,
        )

    def forward(self, x):
        """Do the VAE forward step

        :param x: list of tensors of longs, embed representation of input
        :return: float, kl term component of loss
        :return: float, recon component of loss
        """

        x = lbann.Slice(x, slice_points=str_list([0, self.input_feature_dims]))
        x = lbann.Identity(x)
        x_emb = lbann.Embedding(
            x,
            num_embeddings=self.dictionary_size,
            embedding_dim=self.embedding_size,
            name='emb',
            weights=self.emb_weights
        )

        # Encoder: x -> z, kl_loss
        z, kl_loss = self.forward_encoder(x_emb)

        # Decoder: x, z -> recon_loss
        pred = self.forward_decoder(x_emb, z)
        recon_loss = self.compute_loss(x, pred)

        # Hack to remove blocking GPU allreduce in evaluation layer
        kl_loss = lbann.Identity(kl_loss, device='CPU')
        recon_loss = lbann.Identity(recon_loss, device='CPU')

        return kl_loss, recon_loss

    def forward_encoder(self, x_emb):
        """Encoder step, emulating z ~ E(x) = q_E(z|x)

        :param x_emb: (n_batch, len(x), d_z) of floats, embeddings for input sentence x
        :return: (n_batch, d_z) of floats, sample of latent vector z
        :return: float, kl term component of loss
        """

        # _, h = self.encoder_rnn(x, None)
        h = self.encoder_rnn(x_emb, None)

        h = lbann.Slice(
            h,
            slice_points=str_list([self.input_feature_dims-1,
                                   self.input_feature_dims]),
            axis=0,
        )
        h = lbann.Identity(h)

        mu, logvar = self.q_mu(h), self.q_logvar(h)

        # Set datatype of previous layers
        # Note: Depth-first search from mu and logvar to x_emb
        stack = [mu, logvar]
        in_stack = {l : True for l in stack}
        while stack:
            l = stack.pop()
            if type(l) not in (lbann.Slice, lbann.Reshape, lbann.Tessellate):
                l.datatype = self.datatype
            for parent in l.parents:
                if parent not in in_stack and parent is not x_emb:
                    stack.append(parent)
                    in_stack[parent] = True

        # eps = torch.randn_like(mu)
        eps = lbann.Gaussian(mean=0, stdev=1,hint_layer=mu)

        # z = mu + (logvar / 2).exp() * eps
        z = lbann.Add([mu, (lbann.Multiply([lbann.Exp(lbann.WeightedSum(logvar,scaling_factors='0.5')),eps]))])

        # kl_loss = 0.5 * (logvar.exp() + mu ** 2 - 1 - logvar).sum(1).mean()
        kl_loss = lbann.Reduction(lbann.WeightedSum(
                                        [lbann.Exp(logvar),
                                        lbann.Square(mu),
                                        lbann.Constant(value=1.0, hint_layer=mu),
                                        logvar],
                                        scaling_factors='0.5 0.5 -0.5 -0.5'),
                                        mode='sum')

        return z, kl_loss

    def forward_decoder(self, x_emb, z):
        """Decoder step, emulating x ~ G(z)

        :param x_emb: (n_batch, len(x), d_z) of floats, embeddings for input sentence x
        :param z: (n_batch, d_z) of floats, latent vector z
        :return: float, recon component of loss
        :return: list of ints, reconstructed sentence
        """

        # z_0 = z.unsqueeze(1).repeat(1, x_emb.size(1), 1)
        # x_input = torch.cat([x_emb, z_0], dim=-1)
        z_0 = lbann.Tessellate(
            lbann.Reshape(z, dims=str_list([1, 128])),
            dims=str_list([self.input_feature_dims, 128]),
        )
        x_input = lbann.Concatenation(x_emb, z_0, axis=1)

        h_0 = self.decoder_lat(z)

        # output, _ = self.decoder_rnn(x_input, h_0)
        output = self.decoder_rnn(x_input, [h_0, h_0, h_0])

        # y = self.decoder_fc(output)
        y = lbann.ChannelwiseFullyConnected(
            output,
            output_channel_dims=self.dictionary_size,
            bias=True,
            name=f'{self.decoder_fc.name}',
            weights=self.decoder_fc.weights,
        )

        # Set datatype of layers
        # Note: Depth-first search from y to x_emb and z
        stack = [y]
        in_stack = {l : True for l in stack}
        while stack:
            l = stack.pop()
            if type(l) not in (lbann.Slice, lbann.Reshape, lbann.Tessellate):
                l.datatype = self.datatype
            for parent in l.parents:
                if parent not in in_stack and parent not in (x_emb, z):
                    stack.append(parent)
                    in_stack[parent] = True

        return y

    def compute_loss(self, x, y):

        # y[:, :-1]
        y = lbann.Slice(
            y,
            axis=0,
            slice_points=str_list([0, self.input_feature_dims-1]),
        )
        y = lbann.Identity(y)

        # x[:, 1:]
        x = lbann.Slice(
            x,
            slice_points=str_list([1, self.input_feature_dims]),
        )
        x = lbann.Identity(x)

        # Convert indices in x to one-hot representation
        # Note: Ignored indices result in zero vectors
        ignore_mask = lbann.Equal(
            x,
            lbann.Constant(value=self.label_to_ignore, hint_layer=x),
        )
        keep_mask = lbann.LogicalNot(ignore_mask)
        length = lbann.Reduction(keep_mask, mode='sum')
        length = lbann.Max(
            length,
            lbann.Constant(value=1, num_neurons=str_list([1])),
        )
        x = lbann.Add(
            lbann.Multiply(keep_mask, x),
            lbann.Multiply(ignore_mask, lbann.Constant(value=-1, hint_layer=x)),
        )
        x = lbann.Slice(x, slice_points=str_list(range(self.input_feature_dims)))
        x = [lbann.Identity(x) for _ in range(self.input_feature_dims-1)]
        x = [lbann.OneHot(xi, size=self.dictionary_size) for xi in x]
        x = [lbann.Reshape(xi, dims=str_list([1, self.dictionary_size])) for xi in x]
        x = lbann.Concatenation(x, axis=0)

        # recon_loss = F.cross_entropy(
        #     y[:, :-1].contiguous().view(-1, y.size(-1)),
        #     x[:, 1:].contiguous().view(-1),
        #     ignore_index=self.pad
        # )
        z = lbann.MatMul(
            lbann.Exp(y),
            lbann.Constant(
                value=1,
                num_neurons=str_list([self.dictionary_size, 1]),
            ),
        )
        z = lbann.Log(z)
        z = lbann.MatMul(
            lbann.Reshape(keep_mask, dims=str_list([1, -1])),
            z,
        )
        recon_loss = lbann.MatMul(
            lbann.Reshape(y, dims=str_list([-1, 1])),
            lbann.Reshape(x, dims=str_list([-1, 1])),
            transpose_a=True,
        )
        recon_loss = lbann.Subtract(z, recon_loss)
        recon_loss = lbann.Reshape(recon_loss, dims=str_list([1]))
        recon_loss = lbann.Divide(recon_loss, length)

        return recon_loss
