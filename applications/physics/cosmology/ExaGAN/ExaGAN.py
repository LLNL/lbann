import lbann
import lbann.modules.base
import lbann.models.resnet


class CosmoGAN(lbann.modules.Module):

    global_count = 0  # Static counter, used for default names

    def __init__(self, name=None):
       self.instance = 0
       self.name = (name if name
                     else 'ExaGAN{0}'.format(CosmoGAN.global_count))

       convbnrelu = lbann.models.resnet.ConvBNRelu
       fc = lbann.modules.FullyConnectedModule
       conv = lbann.modules.Convolution2dModule
       bn_stats_grp_sz = 0 #0 global, 1 local

       ##MCR properties #@todo: make multichannel optional
       self.datascale = 4 
       self.linear_scaler=1000.

       self.inits = {'dense': lbann.NormalInitializer(mean=0,standard_deviation=0.02),
                      'conv': lbann.NormalInitializer(mean=0,standard_deviation=0.02), #should be truncated Normal
                      'convT':lbann.NormalInitializer(mean=0,standard_deviation=0.02)}

       self.d1_conv1 = convbnrelu(64, 4,2, 1,False, bn_stats_grp_sz,False,name=self.name+'_disc1_conv1')
       self.d1_conv2 = convbnrelu(128, 4,2,1,False, bn_stats_grp_sz,False,name=self.name+'_disc1_conv2')
       self.d1_conv3 = convbnrelu(256, 4,2,1,False, bn_stats_grp_sz,False,name=self.name+'_disc1_conv3')
       self.d1_conv4 = convbnrelu(512, 4,2,1,False, bn_stats_grp_sz,False,name=self.name+'_disc1_conv4')
       self.d1_fc = fc(1,name=self.name+'_disc1_fc',
                       weights=[lbann.Weights(initializer=self.inits['dense'])])

       #stacked_discriminator, this will be frozen, no optimizer, 
       #layer has to be named for callback
       self.d2_conv1 = convbnrelu(64, 4,2, 1,False, bn_stats_grp_sz,False,name=self.name+'_disc2_conv1')
       self.d2_conv2 = convbnrelu(128, 4,2,1,False, bn_stats_grp_sz,False,name=self.name+'_disc2_conv2')
       self.d2_conv3 = convbnrelu(256, 4,2,1,False, bn_stats_grp_sz,False,name=self.name+'_disc2_conv3')
       self.d2_conv4 = convbnrelu(512, 4,2,1,False, bn_stats_grp_sz,False,name=self.name+'_disc2_conv4')
       self.d2_fc = fc(1,name=self.name+'_disc2_fc',
                       weights=[lbann.Weights(initializer=self.inits['dense'])])
 
       #generator
       self.g_fc1 = fc(32768,name=self.name+'_gen_fc1',
                       weights=[lbann.Weights(initializer=self.inits['dense'])])
       self.g_convT1 = conv(256, 5,stride=2,padding=2,name=self.name+'_gen_convT1',transpose=True,
                       weights=[lbann.Weights(initializer=self.inits['convT'])])
       self.g_convT2 = conv(128, 5,stride=2,padding=2,name=self.name+'_gen_convT2',transpose=True,
                       weights=[lbann.Weights(initializer=self.inits['convT'])])
       self.g_convT3 = conv(64, 5,stride=2,padding=2,name=self.name+'_gen_convT3',transpose=True,
                       weights=[lbann.Weights(initializer=self.inits['convT'])])
       self.g_convT4 = conv(1, 5, stride=2, padding=2, activation=lbann.Tanh,name='gen_img',transpose=True,
                       weights=[lbann.Weights(initializer=self.inits['convT'])])

    def forward(self, img, z):
    #description
        d1_real = self.forward_discriminator1(img)  #instance1
        gen_img = self.forward_generator(z)
        d1_fake = self.forward_discriminator1(lbann.StopGradient(gen_img)) #instance2
        d_adv = self.forward_discriminator2(gen_img) #instance 3 //need to freeze
        #d1s share weights, d1_w is copied to d_adv (through replace weight callback) and freeze
        return d1_real, d1_fake, d_adv,gen_img

    def forward_discriminator1(self,y):
        ch2 = self.inv_transform(lbann.Identity(y))
        y = lbann.Concatenation(lbann.Identity(y),ch2,axis=0)
        img = lbann.Reshape(y, dims='2 128 128')
        x = lbann.LeakyRelu(self.d1_conv1(img), negative_slope=0.2)
        x = lbann.LeakyRelu(self.d1_conv2(x), negative_slope=0.2)
        x = lbann.LeakyRelu(self.d1_conv3(x), negative_slope=0.2)
        x = lbann.LeakyRelu(self.d1_conv4(x), negative_slope=0.2)
        return self.d1_fc(lbann.Reshape(x,dims='32768')) 

    def forward_discriminator2(self,y):
        ch2 = self.inv_transform(lbann.Identity(y))
        y = lbann.Concatenation(lbann.Identity(y),ch2,axis=0)
        img = lbann.Reshape(y, dims='2 128 128')
        x = lbann.LeakyRelu(self.d2_conv1(img), negative_slope=0.2)
        x = lbann.LeakyRelu(self.d2_conv2(x), negative_slope=0.2)
        x = lbann.LeakyRelu(self.d2_conv3(x), negative_slope=0.2)
        x = lbann.LeakyRelu(self.d2_conv4(x), negative_slope=0.2)
        return self.d2_fc(lbann.Reshape(x,dims='32768')) 
 
    def forward_generator(self,z):
        x = lbann.Relu(lbann.BatchNormalization(self.g_fc1(z),decay=0.9,scale_init=1.0,epsilon=1e-5))
        x = lbann.Reshape(x, dims='512 8 8') #channel first
        x = lbann.Relu(lbann.BatchNormalization(self.g_convT1(x),decay=0.9,scale_init=1.0,epsilon=1e-5))
        x = lbann.Relu(lbann.BatchNormalization(self.g_convT2(x),decay=0.9,scale_init=1.0,epsilon=1e-5))
        x = lbann.Relu(lbann.BatchNormalization(self.g_convT3(x),decay=0.9,scale_init=1.0,epsilon=1e-5))
        return self.g_convT4(x) 

    def inv_transform(self,y): 
        inv_transform = lbann.WeightedSum(
                                      lbann.SafeDivide(
                                      lbann.Add(lbann.Constant(value=1.0, hint_layer=y),lbann.Identity(y)),
                                      lbann.Subtract(lbann.Constant(value=1.0, hint_layer=y),lbann.Identity(y))),
                                      scaling_factors=str(self.datascale))
        linear_scale = 1/self.linear_scaler
        CH2 = lbann.Tanh(lbann.WeightedSum(inv_transform,scaling_factors=str(linear_scale)))
        return CH2  
  
