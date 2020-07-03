import lbann
import lbann.modules.base


#Synonymous to fc_gen0
class MACCForward(lbann.modules.Module):

    global_count = 0  # Static counter, used for default names

    #model capacity factor cf
    def __init__(self, out_dim,cf=1,name=None):
       self.instance = 0
       self.name = (name if name
                     else 'macc_forward{0}'.format(MACCForward.global_count))

       fc = lbann.modules.FullyConnectedModule
       
       assert isinstance(cf, int), 'model capacity factor should be an int!'
       #generator #fc2_gen0
       g_neurons = [x*cf for x in [32,256,1024]]
       self.gen_fc = [fc(g_neurons[i],activation=lbann.Relu, name=self.name+'gen_fc'+str(i))
                      for i in range(len(g_neurons))]
       self.predy = fc(out_dim,name=self.name+'pred_out')
      
    def forward(self,x):
        return self.predy(self.gen_fc[2](self.gen_fc[1](self.gen_fc[0](x))))
 
#Synonymous to fc_gen1
class MACCInverse(lbann.modules.Module):

    global_count = 0  # Static counter, used for default names
    #model capacity factor cf
    def __init__(self, out_dim,cf=1,name=None):
       self.instance = 0
       self.name = (name if name
                     else 'macc_inverse{0}'.format(MACCInverse.global_count))

       fc = lbann.modules.FullyConnectedModule
       
       assert isinstance(cf, int), 'model capacity factor should be an int!'
       #generator #fc_gen1
       g_neurons = [x*cf for x in [16,128,64]]
       self.gen_fc = [fc(g_neurons[i],activation=lbann.Relu, name=self.name+'gen_fc'+str(i))
                      for i in range(len(g_neurons))]
       self.predx = fc(out_dim,name=self.name+'pred_out')

    def forward(self,y):
        return self.predx(self.gen_fc[2](self.gen_fc[1](self.gen_fc[0](y))))


class MACCWAE(lbann.modules.Module):

    global_count = 0  # Static counter, used for default names
    #model capacity factor (cf) 
    def __init__(self, encoder_out_dim, decoder_out_dim, scalar_dim = 15, cf=1, use_CNN=False, name=None):
       self.instance = 0
       self.name = (name if name
                     else 'macc_wae{0}'.format(MACCWAE.global_count))

       self.use_CNN = use_CNN

       fc = lbann.modules.FullyConnectedModule
       conv = lbann.modules.Convolution2dModule

       assert isinstance(cf, int), 'model capacity factor should be an int!'

       disc_neurons = [128,64,1]
       encoder_neurons = [x*cf for x in [32,256,128]]
       decoder_neurons = [x*cf for x in [64,128,256]]
       #Enc/Dec sizes  [32, 256, 128]   [64, 128, 256]
       print("CF, Enc/Dec sizes ", cf, " ", encoder_neurons, " ", decoder_neurons) 
       enc_outc = [64,32,16]
       dec_outc = [32,16,4]
       
       #Encoder
       self.enc_fc0 = fc(encoder_neurons[0],activation=lbann.Elu,name=self.name+'_enc_fc0')
       self.enc_fc1 = fc(encoder_neurons[1],activation=lbann.Tanh,name=self.name+'_enc_fc1')
       self.enc_fc2 = fc(encoder_neurons[2],activation=lbann.Tanh,name=self.name+'_enc_fc2')
       self.enc_out = fc(encoder_out_dim,name=self.name+'enc_out')
     
       #Decoder
       self.dec_fc0 = fc(decoder_neurons[0],activation=lbann.Elu,name=self.name+'_dec_fc0')
       self.dec_fc1 = fc(decoder_neurons[1],activation=lbann.Tanh,name=self.name+'_dec_fc1')
       self.dec_fc2 = fc(decoder_neurons[2],activation=lbann.Tanh,name=self.name+'_dec_fc2')
       self.dec_out = fc(decoder_out_dim,name=self.name+'pred_y')
       
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

       #Encoder_CNN
       self.enc_conv = [conv(enc_outc[i], 4, stride=2, padding=1, activation=lbann.Relu,
                        name=self.name+'_enc_conv'+str(i)) for i in range(len(enc_outc))] 

       #Decoder_CNN 
       #Arxiv paper/PNAS configuration is D1: Dense(32,1024)
       self.dec_cnn_fc = fc(16*8*8,activation=lbann.Relu,name=self.name+'_dec_cnn_fc')
       self.dec_fc_sca = fc(scalar_dim, name=self.name+'_dec_sca_fc')
       self.dec_convT = [conv(dec_outc[i], 4, stride=2, padding=1,
                        transpose=True, name=self.name+'_dec_conv'+str(i))
                        for i in range(len(dec_outc))]
 
    def forward(self, z, y):
         
        z_sample = self.encoder(y)

        y_recon = self.decoder(z_sample)

        #d real/fake share weights, shared weights is copied to d_adv 
        #(through replace weight callback) and freeze
        d_real = self.discriminator0(lbann.Concatenation([y,z],axis=0))  
        y_z_sample = lbann.Concatenation([y,z_sample],axis=0)
        d_fake = self.discriminator0(lbann.StopGradient(y_z_sample)) 
        d_adv = self.discriminator1(y_z_sample) #freeze

        return d_real, d_fake, d_adv,y_recon

    def encoder(self, y):
        return self.encoder_cnn(y) if self.use_CNN else self.encoder_fc(y) 

    def encoder_fc(self,y):
        return self.enc_out(self.enc_fc2(self.enc_fc1(self.enc_fc0(y))))

    def encoder_cnn(self,y):
        img_sca = lbann.Slice(y, axis=0, slice_points="0 16384 16399", name=self.name+'_y_slice')
        #assume C first, is data C first?
        img = lbann.Reshape(img_sca, dims='4 64 64',name=self.name+'enc_reshape0')
        x = self.enc_conv[2](self.enc_conv[1](self.enc_conv[0](img)))
        x = lbann.Reshape(x, dims=str(16*8*8), name=self.name+'enc_reshape1')
        h_stack = lbann.Concatenation([x,img_sca],axis=0)
        z = self.enc_out(h_stack)
        return z

    def decoder(self, z):
        return self.decoder_cnn(z) if self.use_CNN else self.decoder_fc(z) 

    def decoder_fc(self,z):
        return self.dec_out(self.dec_fc2(self.dec_fc1(self.dec_fc0(z))))
   
    def decoder_cnn(self,z):
        x = self.dec_cnn_fc(z)
        sca = self.dec_fc_sca(lbann.Identity(x))
        img = lbann.Reshape(lbann.Identity(x), dims="16 8 8", name=self.name+'dec_reshape0')
        img = self.dec_convT[2](lbann.Relu(self.dec_convT[1](lbann.Relu(self.dec_convT[0](img)))))
        #concat for common interface, slice in output
        img = lbann.Reshape(img, dims=str(64*64*4), name=self.name+'dec_reshape1') #?? check tensor shape
        #todo check that concat size == dec_out_dim
        return lbann.Concatenation([img,sca],axis=0)

    def discriminator0(self,input):
        return self.d0_fc2(self.d0_fc1(self.d0_fc0(input)))
        
    def discriminator1(self,input):
        return self.d1_fc2(self.d1_fc1(self.d1_fc0(input)))
