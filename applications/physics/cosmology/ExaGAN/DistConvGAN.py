import lbann
import lbann.modules.base
import lbann.models.resnet



class ConvBNRelu(lbann.modules.Module):
    """Convolution -> Batch normalization -> ReLU

    Adapted from ResNets. Assumes image data in NCDHW format.
    """

    def __init__(self, out_channels, kernel_size, stride, padding,
                 use_bn, bn_zero_init, bn_statistics_group_size,
                 activation, name,
                 conv_weights):
        """Initialize ConvBNRelu module.

        Args:
            out_channels (int): Number of output channels, i.e. number
                of convolution filters.
            kernel_size (int): Size of convolution kernel.
            stride (int): Convolution stride.
            padding (int): Convolution padding.
            use_bn (bool): Whether or not batch normalization layers are used.
            bn_zero_init (bool): Zero-initialize batch normalization
                scale.
            bn_statistics_group_size (int): Aggregation size for batch
                normalization statistics.
            activation (lbann.Layer): The activation function.
            name (str): Module name.
            conv_weights (lbann.Weights): Pre-defined weights.
        """

        super().__init__()
        self.name = name
        self.instance = 0
        self.bn_statistics_group_size = bn_statistics_group_size
        self.activation = activation
        self.use_bn = use_bn
        self.conv_weights = conv_weights

        # Initialize convolution
        self.conv = lbann.modules.Convolution3dModule(
            out_channels, kernel_size,
            stride=stride, padding=padding,
            bias=False, weights=self.conv_weights,
            name=self.name + '_conv')

        # Initialize batch normalization
        if self.use_bn:
            bn_scale_init = 0.0 if bn_zero_init else 1.0
            bn_scale = lbann.Weights(
                initializer=lbann.ConstantInitializer(value=bn_scale_init),
                name=self.name + '_bn_scale')
            bn_bias = lbann.Weights(
                initializer=lbann.ConstantInitializer(value=0.0),
                name=self.name + '_bn_bias')
            self.bn_weights = [bn_scale, bn_bias]

    def forward(self, x):
        self.instance += 1
        layer = self.conv(x)
        if self.use_bn:
            layer = lbann.BatchNormalization(
                layer, weights=self.bn_weights,
                statistics_group_size=self.bn_statistics_group_size,
                decay=0.999,
                name='{0}_bn_instance{1}'.format(
                    self.name, self.instance))
        if self.activation:
            layer = self.activation(
                layer,
                name='{0}_activation_instance{1}'.format(
                    self.name, self.instance))
        return layer

class Deconvolution3dModule(lbann.modules.ConvolutionModule):
    """Basic block for 3D deconvolutional neural networks.

    Applies a deconvolution and a nonlinear activation function.
    This is a wrapper class for ConvolutionModule.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(3, transpose=True, *args, **kwargs)



class Exa3DGAN(lbann.modules.Module):

    global_count = 0  # Static counter, used for default names

    def __init__(self, input_width, input_channel, name=None):
       self.instance = 0
       self.name = (name if name
                     else 'Exa3DGAN{0}'.format(Exa3DGAN.global_count))

       convbnrelu = ConvBNRelu
       fc = lbann.modules.FullyConnectedModule
       conv = lbann.modules.Convolution3dModule

       bn_stats_grp_sz = -1 #0 global, 1 local
       self.input_width = input_width
       self.input_channel = input_channel

       assert self.input_width in [128, 256, 512]

       
       #last_conv_dim = [512,8,8,8] 
       #Use Glorot for conv?
       #initializer=lbann.GlorotUniformInitializer())]
       self.inits = {'dense': lbann.NormalInitializer(mean=0,standard_deviation=0.02),
                      'conv': lbann.NormalInitializer(mean=0,standard_deviation=0.02), #should be truncated Normal
                      'convT':lbann.NormalInitializer(mean=0,standard_deviation=0.02)}
      
       #Discriminator 
       d_channels = [64,128,256,512]
       self.d1_conv = [convbnrelu(d_channels[i], 2, 2, 0, False, bn_stats_grp_sz, False,
                                  name=self.name+'_disc1_conv'+str(i), 
                                  activation=lbann.LeakyRelu,
                                  conv_weights=[lbann.Weights(initializer=self.inits['conv'])])
                   for i in range(len(d_channels))] 
       self.d1_fc = fc(1,name=self.name+'_disc1_fc',
                       weights=[lbann.Weights(initializer=self.inits['dense'])])

       #stacked_discriminator, this will be frozen, no optimizer, 
       #layer has to be named for callback
       self.d2_conv = [convbnrelu(d_channels[i], 2, 2, 0, False, bn_stats_grp_sz, False,
                                  name=self.name+'_disc2_conv'+str(i), 
                                  activation=lbann.LeakyRelu,
                                  conv_weights=[lbann.Weights(initializer=self.inits['conv'])])
                   for i in range(len(d_channels))] 
       self.d2_fc = fc(1,name=self.name+'_disc2_fc',
                       weights=[lbann.Weights(initializer=self.inits['dense'])])
       
       #Generator
       #3D=512*8*8*8, 2D== 512*4*4
       self.g_fc1 = fc(512*8*8*8,name=self.name+'_gen_fc1',
                       weights=[lbann.Weights(initializer=self.inits['dense'])])

       g_channels = [256,128,64]
      
       self.g_convT = [conv(g_channels[i], 2, stride=2, padding=0, transpose=True,
                       weights=[lbann.Weights(initializer=self.inits['convT'])])
                       for i in range(len(g_channels))] 

       self.g_convT3 = conv(input_channel, 2, stride=2, padding=0, activation=lbann.Tanh,name='gen_img',transpose=True,
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
        #@todo: generalize (str_list(in_c, in_w, in_w, (in_w->3D)
        y = lbann.Reshape(y, dims='4 128 128 128',device='CPU')
        x = lbann.LeakyRelu(self.d1_conv[0](y), negative_slope=0.2)
        x = lbann.LeakyRelu(self.d1_conv[1](x), negative_slope=0.2)
        x = lbann.LeakyRelu(self.d1_conv[2](x), negative_slope=0.2)
        x = lbann.LeakyRelu(self.d1_conv[3](x), negative_slope=0.2)
        #@todo, get rid of reshape, infer from conv shape 
        #return self.d1_fc(lbann.Reshape(x,dims='32768',device='CPU')) 
        return self.d1_fc(lbann.Reshape(x,dims='262144',device='CPU')) 

    def forward_discriminator2(self,y):
        #@todo: generalize (str_list(in_c, in_w, in_w, (in_w->3D)
        y = lbann.Reshape(y, dims='4 128 128 128', name='d2_in_reshape', device='CPU')
        x = lbann.LeakyRelu(self.d2_conv[0](y), negative_slope=0.2)
        x = lbann.LeakyRelu(self.d2_conv[1](x), negative_slope=0.2)
        x = lbann.LeakyRelu(self.d2_conv[2](x), negative_slope=0.2)
        x = lbann.LeakyRelu(self.d2_conv[3](x), negative_slope=0.2)
        #return self.d2_fc(lbann.Reshape(x,dims='32768',name='d2_out_reshape', device='CPU')) 
        #@todo, get rid of reshape, infer from conv shape 
        return self.d2_fc(lbann.Reshape(x,dims='262144',name='d2_out_reshape', device='CPU')) 
 
    def forward_generator(self,z):
        x = lbann.Relu(lbann.BatchNormalization(self.g_fc1(z),decay=0.9,scale_init=1.0,epsilon=1e-5, device='CPU',),device='CPU')
        #x = lbann.Reshape(x, dims='512 8 8') #channel first
        x = lbann.Reshape(x, dims='512 8 8 8',name='gen_zin_reshape', device='CPU') #new
        x = lbann.Relu(lbann.BatchNormalization(self.g_convT[0](x),decay=0.9,scale_init=1.0,epsilon=1e-5))
        x = lbann.Relu(lbann.BatchNormalization(self.g_convT[1](x),decay=0.9,scale_init=1.0,epsilon=1e-5))
        x = lbann.Relu(lbann.BatchNormalization(self.g_convT[2](x),decay=0.9,scale_init=1.0,epsilon=1e-5))
        return self.g_convT3(x) 

