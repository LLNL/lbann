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
            num_neurons=str_list([num_layers, hidden_size]),
            name=f'{self.name}_zeros',
            device=self.device,
            datatype=self.datatype,
        )

    def forward(self, x, h=None):
        self.instance += 1
        if not h:
            h = self.zeros
        y = lbann.GRU(
            x,
            h,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            name=f'{self.name}_instance{self.instance}',
            weights=self.weights,
            device=self.device,
            datatype=self.datatype,
        )
            #if(i < self.num_layers-1):
            #    x = lbann.Dropout(x, keep_prob=0.5)
        return y

class MolWAE(lbann.modules.Module):
    """Molecular WAE.

    """

    global_count = 0  # Static counter, used for default names

    def __init__(self, input_feature_dims,dictionary_size, embedding_size,
                 ignore_label,num_decoder_layers=3,save_output=False, name=None):
        """Initialize Molecular WAE.

        Args:
            input_feature_dims (int): analogous to sequence length.
            dictionary_size (int): vocabulary size
            embedding_size (int): embedding size
            ignore_label (int): padding index
            num_decoder_layers (int, optional) : Number of decoder layers
                (default: 3)
            save_output (bool, optional): save or not save predictions
                (default: False).
            name (str, optional): Module name
                (default: 'molvae_module<index>').

        """
        MolWAE.global_count += 1
        self.instance = 0
        self.name = (name if name
                     else 'molvae_module{0}'.format(MolWAE.global_count))

        self.input_feature_dims = input_feature_dims
        self.embedding_size = embedding_size
        self.dictionary_size = dictionary_size
        self.label_to_ignore = ignore_label
        self.num_decoder_layers = num_decoder_layers
        self.save_output = save_output
        self.datatype = lbann.DataType.FLOAT
        self.weights_datatype = lbann.DataType.FLOAT

        fc = lbann.modules.FullyConnectedModule
        gru = GRUModule

        disc_neurons = [128,64,1]
        #Encoder
        self.encoder_rnn = gru(
            hidden_size=256,
            name=self.name+'_encoder_rnn',
            datatype=self.datatype,
            weights_datatype=self.weights_datatype,
        )
        self.q_mu = fc(128,name='encoder_qmu')
        self.q_logvar = fc(128,name='encoder_qlogvar')
        for w in self.q_mu.weights + self.q_logvar.weights:
            w.datatype = self.weights_datatype

        #Decoder
        self.decoder_rnn = gru(
            hidden_size=512,
            num_layers=self.num_decoder_layers,
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

        #Discriminator1
        self.d0_fc0 = fc(disc_neurons[0],activation=lbann.Relu,name=self.name+'_disc0_fc0')
        self.d0_fc1 = fc(disc_neurons[1],activation=lbann.Relu,name=self.name+'_disc0_fc1')
        self.d0_fc2 = fc(disc_neurons[2],name=self.name+'_disc0_fc2')

        #Discriminator2
        #stacked_discriminator, this will be frozen, no optimizer,
        #layer has to be named for replace layer callback
        self.d1_fc0 = fc(disc_neurons[0],activation=lbann.Relu,name=self.name+'_disc1_fc0')
        self.d1_fc1 = fc(disc_neurons[1],activation=lbann.Relu,name=self.name+'_disc1_fc1')
        self.d1_fc2 = fc(disc_neurons[2],name=self.name+'_disc1_fc2')

    def forward(self, x, z):
        """Do the WAE forward step

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
        z_sample = self.forward_encoder(x_emb)

        # Decoder: x, z -> recon_loss
        #pred = self.forward_decoder(x_emb, z_sample)
        pred, arg_max = self.forward_decoder(x_emb, z_sample)
        recon_loss = self.compute_loss(x, pred)

        # Hack to remove blocking GPU allreduce in evaluation layer
        #kl_loss = lbann.Identity(kl_loss, device='CPU')
        recon_loss = lbann.Identity(recon_loss, device='CPU')

        z_prior = lbann.Tessellate(
            lbann.Reshape(z, dims=str_list([1, 128])),
            dims=str_list([self.input_feature_dims, 128]),
        )

        d_real = self.discriminator0(lbann.Concatenation([x_emb,z_prior],axis=1))

        z_sample0 = lbann.Tessellate(
            lbann.Reshape(z_sample, dims=str_list([1, 128])),
            dims=str_list([self.input_feature_dims, 128]),
        )
        y_z_sample = lbann.Concatenation([x_emb,z_sample0],axis=1)

        d_fake = self.discriminator0(lbann.StopGradient(y_z_sample))
        d_adv = self.discriminator1(y_z_sample) #freeze

        return recon_loss, d_real, d_fake, d_adv, arg_max

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
        z = self.q_mu(h)
        return z

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
        # h_0 = h_0.unsqueeze(0).repeat(self.decoder_rnn.num_layers, 1, 1)
        h_0 = lbann.Reshape(h_0, dims=str_list([1, 512]))
        h_0 = lbann.Tessellate(h_0, dims=str_list((self.num_decoder_layers, 512)))

        # output, _ = self.decoder_rnn(x_input, h_0)
        output = self.decoder_rnn(x_input, h_0)

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

        # Find argmax
        if(self.save_output):
          y_slice = lbann.Slice(
              y,
              axis=0,
              slice_points=str_list(range(self.input_feature_dims+1)),
          )
          y_slice = [lbann.Reshape(y_slice, dims='-1') for _ in range(self.input_feature_dims)]
          arg_max = [lbann.Argmax(yi, device='CPU') for yi in y_slice]

          return y, arg_max
        else:
          return y, None

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
            self.constant(self.label_to_ignore, hint_layer=x),
        )
        keep_mask = lbann.LogicalNot(ignore_mask)
        length = lbann.Reduction(keep_mask, mode='sum')
        length = lbann.Max(length, self.constant(1, [1]))
        x = lbann.Add(
            lbann.Multiply(keep_mask, x),
            lbann.Multiply(ignore_mask, self.constant(-1, hint_layer=x)),
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
        # Note: Ideally we'd shift y by y.max(-1) for numerical stability
        shifts = lbann.MatMul(
            lbann.Max(y, self.constant(0, hint_layer=y)),
            self.constant(
                1 / math.sqrt(self.dictionary_size),
                [self.dictionary_size, self.dictionary_size],
            ),
        )
        y = lbann.Subtract(y, shifts)
        z = lbann.MatMul(
            lbann.Exp(y),
            self.constant(1, [self.dictionary_size, 1]),
        )
        z = lbann.Log(z)
        z = lbann.MatMul(
            lbann.Reshape(keep_mask, dims=str_list([1, -1])),
            z,
        )
        recon_loss = lbann.MatMul(
            lbann.Reshape(y, dims=str_list([1, -1])),
            lbann.Reshape(x, dims=str_list([1, -1])),
            transpose_b=True,
        )
        recon_loss = lbann.Subtract(z, recon_loss)
        recon_loss = lbann.Reshape(recon_loss, dims=str_list([1]))
        recon_loss = lbann.Divide(recon_loss, length)

        return recon_loss

    def constant(self, value, dims=[], datatype=None, hint_layer=None):
        return lbann.Constant(
            value=value,
            num_neurons=str_list(dims),
            datatype=datatype,
            hint_layer=hint_layer,
        )

    def discriminator0(self,input):
        return self.d0_fc2(self.d0_fc1(self.d0_fc0(input)))

    def discriminator1(self,input):
        return self.d1_fc2(self.d1_fc1(self.d1_fc0(input)))
