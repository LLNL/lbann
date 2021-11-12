import lbann
import lbann.modules.base


class WAE(lbann.modules.Module):

    global_count = 0  # Static counter, used for default names

    def __init__(self, encoder_out_dim, decoder_out_dim, name=None):
       self.instance = 0
       self.name = (name if name
                     else 'wae{0}'.format(WAE.global_count))

       fc = lbann.modules.FullyConnectedModule
       disc_neurons = [128,64,1]
       encoder_neurons = [32,256,128]
       decoder_neurons = [64,128,256]
       
       #Encoder
       self.enc_fc0 = fc(encoder_neurons[0],activation=lbann.Elu,name=self.name+'_enc_fc0')
       self.enc_fc1 = fc(encoder_neurons[1],activation=lbann.Tanh,name=self.name+'_enc_fc1')
       self.enc_fc2 = fc(encoder_neurons[2],activation=lbann.Tanh,name=self.name+'_enc_fc2')
       self.enc_out = fc(encoder_out_dim,name='enc_out')
       
       #Decoder
       self.dec_fc0 = fc(decoder_neurons[0],activation=lbann.Elu,name=self.name+'_dec_fc0')
       self.dec_fc1 = fc(decoder_neurons[1],activation=lbann.Tanh,name=self.name+'_dec_fc1')
       self.dec_fc2 = fc(decoder_neurons[2],activation=lbann.Tanh,name=self.name+'_dec_fc2')
       self.dec_out = fc(decoder_out_dim,name='pred_y')
       
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
 

    def forward(self, z, y):
    
        z_sample = self.forward_encoder(y)

        y_recon = self.forward_decoder(z_sample)

        #d real/fake share weights, shared weights is copied to d_adv 
        #(through replace weight callback) and freeze
        d_real = self.forward_discriminator0(lbann.Concatenation([y,z],axis=0))  
        y_z_sample = lbann.Concatenation([y,z_sample],axis=0)
        d_fake = self.forward_discriminator0(lbann.StopGradient(y_z_sample)) 
        d_adv = self.forward_discriminator1(y_z_sample) #freeze

        return d_real, d_fake, d_adv,y_recon

    def forward_encoder(self,y):
        bn = lbann.BatchNormalization
        return self.enc_out(bn(self.enc_fc2(bn(self.enc_fc1(bn(self.enc_fc0(y),epsilon=1e-3)
                               ),epsilon=1e-3)),epsilon=1e-3))

    def forward_decoder(self,z):
        return self.dec_out(self.dec_fc2(self.dec_fc1(self.dec_fc0(z))))

    def forward_discriminator0(self,input):
        return self.d0_fc2(self.d0_fc1(self.d0_fc0(input)))
        
    def forward_discriminator1(self,input):
        return self.d1_fc2(self.d1_fc1(self.d1_fc0(input)))
