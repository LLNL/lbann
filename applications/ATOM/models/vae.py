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
        gru = lbann.modules.GRU
        #Encoder
        winit = lbann.GlorotNormalInitializer()
        self.encoder_rnn = gru(size=256, name=self.name+'_encoder_rnn')
        self.q_mu = fc(128,name=self.name+'_encoder_qmu')
        self.q_logvar = fc(128,name=self.name+'_encoder_qlogvar')
        #Decoder
        self.decoder_rnn0 = gru(size=512, name=self.name+'_decoder_rnn0')
        self.decoder_rnn1 = gru(size=512, name=self.name+'_decoder_rnn1')
        self.decoder_rnn2 = gru(size=512, name=self.name+'_decoder_rnn2')
        self.decoder_lat = fc(512,name=self.name+'_decoder_lat')
        self.decoder_fc = fc(dictionary_size,name=self.name+'_decoder_fc')
        #shared encoder/decodeer weights
        self.emb_weights = lbann.Weights(initializer=lbann.NormalInitializer(mean=0, standard_deviation=1),
                                   name='emb_matrix')

    def forward(self, x):
        """Do the VAE forward step

        :param x: list of tensors of longs, embed representation of input
        :return: float, kl term component of loss
        :return: float, recon component of loss
        """

        emb = lbann.Embedding(x,
                              num_embeddings=self.dictionary_size,
                              embedding_dim=self.embedding_size,
                              name='emb',
                              weights=self.emb_weights)
        emb_slice = lbann.Slice(emb,
                                axis=0,
                                slice_points=str_list(range(self.input_feature_dims+1)),
                                name='emb_slice')
        emb_list = [lbann.Reshape(emb_slice, dims='-1', name='emb'+str(i))
                    for i in range(self.input_feature_dims)]

        # Encoder: x -> z, kl_loss
        z, kl_loss = self.forward_encoder(emb_list)

        # Decoder: x, z -> recon_loss
        recon_loss, arg_max = self.forward_decoder(x, emb_list, z)

        return kl_loss, recon_loss, arg_max

    def forward_encoder(self, emb_list):
        """Encoder step, emulating z ~ E(x) = q_E(z|x)

        :param embed_list: list of tensors of floats, input sentence emb_list
        :return: (n_batch, d_z) of floats, sample of latent vector z
        :return: float, kl term component of loss
        """

        h = lbann.Constant(value=0.0, num_neurons='256')
        for i in range(self.input_feature_dims):
            _, h = self.encoder_rnn(emb_list[i], h)

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

    def forward_decoder(self, x, emb_list, z):
        """Decoder step, emulating x ~ G(z)

        :param x: list of tensors of longs, input sentence x
        :param emb_list: embeddings of x
        :param z: (n_batch, d_z) of floats, latent vector z
        :return: float, recon component of loss
        """

        # x[:, 1:]
        xshift = lbann.Slice(x, slice_points=str_list([1, self.input_feature_dims]))
        xshift = lbann.Identity(xshift)
        xshift_slice = lbann.Slice(xshift, slice_points=str_list(range(self.input_feature_dims)))
        xshift_list = [lbann.Identity(xshift_slice) for i in range(self.input_feature_dims-1)]

        # Unroll RNN
        h = [self.decoder_lat(z)] * 3
        recon_loss = []
        arg_max = []
        for i in range(self.input_feature_dims-1):

            # RNN stack
            x_input = lbann.Concatenation(emb_list[i], z)
            _, h[0] = self.decoder_rnn0(x_input, h[0])
            _, h[1] = self.decoder_rnn1(h[0], h[1])
            _, h[2] = self.decoder_rnn2(h[1], h[2])
            output = h[2]
            #output = h[0]
            y = self.decoder_fc(output)
            arg_max.append(lbann.Argmax(y,device='CPU'))

            # Cross entropy loss
            y = lbann.Softmax(y)
            xshift_onehot = lbann.OneHot(xshift_list[i], size=self.dictionary_size)
            recon_loss.append(lbann.CrossEntropy(y, xshift_onehot))

        # Average cross entropy over sequence length
        pad_mask = lbann.NotEqual(xshift,
                                  lbann.Constant(value=self.label_to_ignore, hint_layer=xshift))
        length = lbann.Reduction(pad_mask, mode='sum')
        length = lbann.Max(length, lbann.Constant(value=1, num_neurons="1"))
        recon_loss = lbann.Concatenation(recon_loss)
        recon_loss = lbann.Multiply(recon_loss, pad_mask)
        recon_loss = lbann.Reduction(recon_loss, mode='sum')
        recon_loss = lbann.Divide(recon_loss, length)

        return recon_loss, arg_max
