import lbann
import lbann.modules
from math import sqrt
from lbann.util import make_iterable

def _str_list(l):
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
            output_size (int): Size of output tensor.
            name (str, optional): Module name
                (default: 'lenet_module<index>').

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
        #lstm = lbann.modules.LSTMCell
        gru = lbann.modules.GRU
        #Encoder
        winit = lbann.GlorotNormalInitializer()
        self.encoder_rnn = gru(size=256, name=self.name+'_encoder_rnn')
        self.q_mu = fc(128,name=self.name+'q_mu')
        #               weights=[lbann.Weights(initializer=winit)])
        self.q_logvar = fc(128,name=self.name+'q_logvar')
        #                  weights=[lbann.Weights(initializer=winit)])
        #Decoder 
        self.decoder_rnn0 = gru(size=512, name=self.name+'_decoder_rnn0')
        #self.decoder_rnn1 = gru(size=512, name=self.name+'_decoder_rnn1')
        #self.decoder_rnn2 = gru(size=512, name=self.name+'_decoder_rnn2')
        self.decoder_lat = fc(512,name=self.name+'decoder_lat')
        #                      weights=[lbann.Weights(initializer=winit)])
        self.decoder_fc = fc(dictionary_size,name='decoder_fc')
        #                      weights=[lbann.Weights(initializer=winit)])
        #shared encoder/decodeer weights
        #self.emb_weights = lbann.Weights(initializer=lbann.ConstantInitializer(),
        self.emb_weights = lbann.Weights(initializer=lbann.NormalInitializer(mean=0, standard_deviation=1),
                                   name='emb_matrix')
        self.enc_rnn_last_output = lbann.Constant(value=0.0, num_neurons='256')

    def forward(self, x):
        """Do the VAE forward step

        :param x: list of tensors of longs, embed representation of input 
        :return: float, kl term component of loss
        :return: float, recon component of loss
        """

        # Encoder: x -> z, kl_loss
        #z, kl_loss = self.forward_encoder(x)
        #input
        #slice
        #x= identity1
        #x2= identity2
        #create all identity list to feed next step
        idl = []
        for i in range(self.input_feature_dims):
          idl.append(lbann.Identity(x, name='slice_idl_'+str(i)))
       
          
        kl = []
        recon = []
        arg_max = []
        for i in range(self.input_feature_dims):
          #idl = lbann.Identity(x, name='slice_idl_'+str(i))
          emb_l = lbann.Embedding(idl[i],num_embeddings=self.dictionary_size,
                                 embedding_dim=self.embedding_size,name='emb_'+str(i), 
                                 weights=self.emb_weights)
 
          z,kl_loss = self.forward_encoder(emb_l)
          #z = self.forward_encoder(emb_l)
          y_soft_arg = recon_loss  = lbann.Constant(value=0, num_neurons='1') 
        # Decoder: x, z -> recon_loss
          #y_soft_arg, recon_loss = [self.forward_decoder(idl[i],idl[i+1],z) if i < self.input_feature_dims-1 ]
          #               else [const_layer,const_layer]]
          #recon_loss = self.forward_decoder(idl[i],next_x, z)
          if(i < self.input_feature_dims-1):
            y_soft_arg, recon_loss = self.forward_decoder(idl[i],idl[i+1],z)

          kl.append(kl_loss)
          recon.append(recon_loss)
          #loss.append(lbann.LayerTerm(ce_mask, scale=1/(sequence_length-1)))
          #recon.append(lbann.LayerTerm(recon_loss, scale=1/(self.input_feature_dims-1)))
          arg_max.append(y_soft_arg)

        return kl, recon, arg_max
        #return recon, arg_max

    def forward_encoder(self, x):
        """Encoder step, emulating z ~ E(x) = q_E(z|x)

        :param x: list of tensors of longs, input sentence x
        :return: (n_batch, d_z) of floats, sample of latent vector z
        :return: float, kl term component of loss
        """
        '''
        x = [self.x_emb(i_x) for i_x in x]
        x = nn.utils.rnn.pack_sequence(x)
        '''
        #last_output = lbann.Constant(value=0.0, num_neurons='256')
        #last_cell = lbann.Constant(value=0.0, num_neurons='256',
        #                                name=self.name + '_init_cell')
        
        #prev_state = [last_output, last_cell]
        #prev_state = last_output
         
        #h = self.encoder_rnn(x, [prev_state])
        x,self.enc_rnn_last_output = self.encoder_rnn(x, [self.enc_rnn_last_output])

          #h = h[-(1 + int(self.encoder_rnn.bidirectional)):]
          #h = torch.cat(h.split(1), dim=-1).squeeze(0)

          #mu, logvar = self.q_mu(h), self.q_logvar(h)
        #mu = self.q_mu(h[0])
        #mu = self.q_mu(x)
        #KL from here
        #mu, logvar = self.q_mu(h[0]), self.q_logvar(h[0])
        mu, logvar = self.q_mu(x), self.q_logvar(x)
          #eps = torch.randn_like(mu)
        
        eps = lbann.Gaussian(mean=0, stdev=1,hint_layer=mu) 
                         # name='{0}_eps_instance{1}'.format(self.name,self.instance))
          #z = mu + (logvar / 2).exp() * eps
        z = lbann.Add([mu, (lbann.Multiply([lbann.Exp(lbann.WeightedSum(logvar,scaling_factors='0.5')),eps]))])

         #kl_loss = 0.5 * (logvar.exp() + mu ** 2 - 1 - logvar).sum(1).mean()
        kl_loss = lbann.Reduction(lbann.WeightedSum(
                                        [lbann.Exp(logvar),
                                        lbann.Square(mu),
                                        lbann.Constant(value=1.0, hint_layer=mu),
                                        logvar], 
                                        scaling_factors='0.5 0.5 -0.5 -0.5'),
                                        mode='sum')
         

        return z, kl_loss
        #return mu

    def forward_decoder(self, x,next_x, z):
        """Decoder step, emulating x ~ G(z)

        :param x: list of tensors of longs, input sentence x
        :param z: (n_batch, d_z) of floats, latent vector z
        :return: float, recon component of loss
        """
        '''
        lengths = [len(i_x) for i_x in x]

        x = nn.utils.rnn.pad_sequence(x, batch_first=True,
                                      padding_value=self.pad)
        x_emb = self.x_emb(x)

        z_0 = z.unsqueeze(1).repeat(1, x_emb.size(1), 1)
        x_input = torch.cat([x_emb, z_0], dim=-1)
        x_input = nn.utils.rnn.pack_padded_sequence(x_input, lengths,
                                                    batch_first=True)
        '''
          
        x_emb = lbann.Embedding(x,num_embeddings=self.dictionary_size,
                                 embedding_dim=self.embedding_size,
                                 weights=self.emb_weights)

        x_input = lbann.Concatenation([lbann.Reshape(x_emb,dims=str(self.embedding_size)), z])
        h_0 = self.decoder_lat(z)
        #h_0 = h_0.unsqueeze(0).repeat(self.decoder_rnn.num_layers, 1, 1)

        #output, _ = self.decoder_rnn(x_input, h_0)
        #last_output = lbann.Constant(value=0.0, num_neurons='256',
        #                                  name=self.name + '_decinit_output')
        #last_cell = lbann.Constant(value=0.0, num_neurons='256',
         #                               name=self.name + '_decinit_cell')
        
        #prev_state = [last_output, last_cell]
        output, _ = self.decoder_rnn0(x_input, [h_0])
        #out,_ = self.decoder_rnn0(x_input, [h_0])
        #out,state = self.decoder_rnn1(out, [state])
        #out,_ = self.decoder_rnn1(out, [h_0])
        #output, _ = self.decoder_rnn2(out, [h_0])

        #output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        y = self.decoder_fc(output)
        y_soft = lbann.Softmax(y)
        y_soft_arg = lbann.Argmax(y_soft,device='CPU')
        #recon_loss = F.cross_entropy(
        #    y[:, :-1].contiguous().view(-1, y.size(-1)),
        #    x[:, 1:].contiguous().view(-1),
        #    ignore_index=self.pad
        #)
        gt = lbann.OneHot(next_x, size=self.dictionary_size)
        ce = lbann.CrossEntropy([y_soft, gt])
        # mask padding in input
        pad_mask = lbann.NotEqual(
            [x, lbann.Constant(value=self.label_to_ignore, num_neurons="1")],
        )  
        recon_loss = lbann.Multiply([pad_mask, ce])
        #loss.append(lbann.LayerTerm(ce_mask, scale=1 / (sequence_length - 1)))
        #recon_loss = lbann.CrossEntropy([y,x], name='recon_loss')
        #recon_loss = lbann.MeanSquaredError([y,x], name='recon_loss')
        '''
        recon_loss = lbann.CrossEntropy(
                                      [y_soft,
                                       lbann.LabelToVec(next_x,num_neurons=str(self.dictionary_size),
                                                        label_to_ignore=self.label_to_ignore,
                                                         )]
                                        )

        '''
        return y_soft_arg, recon_loss
