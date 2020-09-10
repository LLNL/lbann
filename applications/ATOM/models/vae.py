import lbann
import lbann.modules
from math import sqrt
from lbann.util import make_iterable

def str_list(l):
    """Convert an iterable object to a space-separated string."""
    return ' '.join(str(i) for i in make_iterable(l))

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

        fc = lbann.modules.FullyConnectedModule
        #Encoder
        self.q_mu = fc(128,name=self.name+'_qmu')
        self.q_logvar = fc(128,name=self.name+'_qlogvar')
        #Decoder
        self.decoder_lat = fc(512,name=self.name+'_decoder_lat')
        #shared encoder/decodeer weights
        self.emb_weights = lbann.Weights(initializer=lbann.NormalInitializer(mean=0, standard_deviation=1),
                                   name='emb_matrix')

    def forward(self, x):
        """Do the VAE forward step

        :param x: list of tensors of longs, embed representation of input
        :return: float, kl term component of loss
        :return: float, recon component of loss
        """

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
        pred, arg_max = self.forward_decoder(x_emb, z)
        recon_loss = self.compute_loss(x, pred)

        return kl_loss, recon_loss, arg_max

    def forward_encoder(self, x_emb):
        """Encoder step, emulating z ~ E(x) = q_E(z|x)

        :param x_emb: (n_batch, len(x), d_z) of floats, embeddings for input sentence x
        :return: (n_batch, d_z) of floats, sample of latent vector z
        :return: float, kl term component of loss
        """

        h = lbann.Constant(value=0.0, num_neurons='256')
        h = lbann.GRU(
            x_emb, h,
            hidden_size=256,
            name=f'{self.name}_encoder_rnn',
        )
        h = lbann.Slice(
            h,
            slice_points=str_list([self.input_feature_dims-1,
                                   self.input_feature_dims]),
            axis=0,
        )
        h = lbann.Identity(h)

        mu, logvar = self.q_mu(h), self.q_logvar(h)

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
        h_1 = lbann.GRU(
            x_emb, h_0,
            hidden_size=512,
            name=f'{self.name}_decoder_rnn0',
        )
        h_2 = lbann.GRU(
            h_1, h_0,
            hidden_size=512,
            name=f'{self.name}_decoder_rnn1',
        )
        output = lbann.GRU(
            h_2, h_0,
            hidden_size=512,
            name=f'{self.name}_decoder_rnn2',
        )

        # y = self.decoder_fc(output)
        y = lbann.ChannelwiseFullyConnected(
            output,
            output_channel_dims=self.dictionary_size,
            bias=True,
            name=f'{self.name}_decoder_fc',
        )

        # Find argmax
        y_slice = lbann.Slice(
            y,
            axis=0,
            slice_points=str_list(range(self.input_feature_dims+1)),
        )
        y_slice = [lbann.Reshape(y_slice, dims='-1') for _ in range(self.input_feature_dims)]
        arg_max = [lbann.Argmax(yi, device='CPU') for yi in y_slice]

        return y, arg_max

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

        # Set ignored labels to -1
        ignore_mask = lbann.Equal(
            x,
            lbann.Constant(value=self.label_to_ignore, hint_layer=x),
        )
        keep_mask = lbann.LogicalNot(ignore_mask)
        length = lbann.Reduction(keep_mask, mode='sum')
        length = lbann.Max(length, lbann.Constant(value=1, num_neurons="1"))
        x = lbann.Add(
            lbann.Multiply(keep_mask, x),
            lbann.Multiply(ignore_mask, lbann.Constant(value=-1, hint_layer=x)),
        )

        # recon_loss = F.cross_entropy(
        #     y[:, :-1].contiguous().view(-1, y.size(-1)),
        #     x[:, 1:].contiguous().view(-1),
        #     ignore_index=self.pad
        # )
        y = lbann.ChannelwiseSoftmax(y)
        x = lbann.Slice(x, slice_points=str_list(range(self.input_feature_dims)))
        x = [lbann.Identity(x) for _ in range(self.input_feature_dims-1)]
        x = [lbann.OneHot(xi, size=self.dictionary_size) for xi in x]
        x = [lbann.Reshape(xi, dims=str_list([1, self.dictionary_size])) for xi in x]
        x = lbann.Concatenation(x, axis=0)
        recon_loss = lbann.CrossEntropy(y, x)
        recon_loss = lbann.Divide(recon_loss, length)

        return recon_loss
