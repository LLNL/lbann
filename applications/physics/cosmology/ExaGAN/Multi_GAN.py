import lbann
import lbann.modules.base
import lbann.models.resnet
import lbann.modules as lm
from lbann.core.util import get_parallel_strategy_args
import numpy as np


class ConvBNRelu(lbann.modules.Module):
    """Convolution -> Batch normalization -> ReLU

    Adapted from ResNets. Assumes image data in NCDHW format.
    """

    def __init__(self, out_channels, kernel_size, stride, padding,
                 use_bn, bn_zero_init, bn_statistics_group_size,
                 activation, parallel_strategy, name,
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
        self.ps = parallel_strategy

        # Initialize convolution
        self.conv = lbann.modules.Convolution3dModule(
            out_channels, kernel_size,
            stride=stride, padding=padding,
            bias=False, parallel_strategy=self.ps, 
            weights=self.conv_weights,
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
                parallel_strategy=self.ps,
                name='{0}_bn_instance{1}'.format(
                    self.name, self.instance))
        if self.activation:
            layer = self.activation(
                layer,
                parallel_strategy=self.ps,
                name='{0}_activation_instance{1}'.format(
                    self.name, self.instance))
        return layer



class Exa3DMultiGAN(lbann.modules.Module):
    """(Conditional) Generative model for 3D cosmology.
        It is not conditional by default. For conditional GAN, call forward method with labels (sigma parameters) 

        Args:
            input_width (int): Size of input image i.e., one of 3D spatial dimensions (default=64)
            input_channel (int): Number of input channels (default=1).
            gen_device (str): Device allocation for generator network (default=GPU).
            disc_ps (lbann.ParallelStrategy): Parallel strategy for discriminator network (default=None).
            gen_ps (lbann.ParallelStrategy): Parallel strategy for generator network (default=None).
            use_bn (bool): Whether or not batch normalization layers are used (default= False).
            enable_subgraph (bool): Whether or not use subgraph parallelism in discriminator layers (default= False).
            name (str): Module name.
    """

    global_count = 0  # Static counter, used for default names

    def __init__(self, input_width=64, input_channel=1,gen_device='GPU',
                 disc_ps=None, gen_ps=None,use_bn=False,num_discblocks=1,
                 enable_subgraph=False,name=None, alternate_updates=False):

       self.instance = 0
       self.name = (name if name
                     else 'Exa3DMultiGAN{0}'.format(Exa3DMultiGAN.global_count))
       convbnrelu = ConvBNRelu
       fc = lbann.modules.FullyConnectedModule
       conv = lbann.modules.Convolution3dModule
       bn_stats_grp_sz = -1 #0 global, 1 local
       self.input_width = input_width
       self.input_channel = input_channel

       self.g_device = gen_device
       #Set parallel strategy
       self.d_ps = disc_ps 
       self.g_ps = gen_ps 
       self.use_bn = use_bn
       self.enable_subgraph = enable_subgraph

       self.alternate_updates = alternate_updates

       assert self.input_width in [64,128, 256, 512]

       w = [int(self.input_width/16)]*3 #filter size in last disc conv and first gen conv 
       w.insert(0,512) ##num filters in last disc conv and first gen conv
       self.outc_dims = w
      


       self.inits = {'dense': lbann.NormalInitializer(mean=0,standard_deviation=0.02),
                      'conv': lbann.NormalInitializer(mean=0,standard_deviation=0.02), 
                      'convT':lbann.NormalInitializer(mean=0,standard_deviation=0.02)}
      
       #Discriminator 
       d_channels = [64,128,256,512]
       kernel_size=5
       padding = 2
       stride = 2
       self.num_blocks = num_discblocks
       #todo add bias
       self.d1_conv = [[convbnrelu(d_channels[i], kernel_size, stride, padding, self.use_bn, bn_stats_grp_sz, False,
                                  name=self.name+'_disc1_conv'+str(i)+'b'+str(b), 
                                  activation=lbann.Relu,
                                  parallel_strategy=self.d_ps,
                                  conv_weights=[lbann.Weights(initializer=self.inits['conv'],name=self.name+'_d1_convw'+str(i)+'b'+str(b))])
                   for i in range(len(d_channels))] for b in range(self.num_blocks)] 

       self.d1_fc = [fc(1,name=self.name+'_disc1_fc_b'+str(b),
                       weights=[lbann.Weights(initializer=self.inits['dense'], name=self.name+'_d1_fcw_b'+str(b))])
                     for b in range(self.num_blocks)]

       if not self.alternate_updates:
            #stacked_discriminator, this will be frozen, no optimizer, 
            #layer has to be named for callback
            self.d2_conv = [[convbnrelu(d_channels[i], kernel_size, stride, padding, self.use_bn, bn_stats_grp_sz, False,
                                        name=self.name+'_disc2_conv'+str(i)+'b'+str(b), 
                                        activation=lbann.Relu,
                                        parallel_strategy=self.d_ps,
                                        conv_weights=[lbann.Weights(initializer=self.inits['conv'],name=self.name+'_d2_convw'+str(i)+'b'+str(b))])
                        for i in range(len(d_channels))] for b in range(self.num_blocks)] 

            self.d2_fc = [fc(1,name=self.name+'_disc2_fc_b'+str(b),
                            weights=[lbann.Weights(initializer=self.inits['dense'], name=self.name+'_d2_fcw_b'+str(b))])
                            for b in range(self.num_blocks)]
        

       g_channels = [256,128,64]
       kernel_size=4
       padding=1
       self.g_convT = [conv(g_channels[i], kernel_size, stride, padding, transpose=True,
                        name=self.name+'_gen'+str(i),
                        parallel_strategy=self.g_ps,
                       weights=[lbann.Weights(initializer=self.inits['convT'])])
                       for i in range(len(g_channels))] 

       self.g_convT3 = conv(1, kernel_size, stride, padding, activation=lbann.Tanh,
                            parallel_strategy=self.g_ps,
                            name='gen_img',transpose=True,
                            weights=[lbann.Weights(initializer=self.inits['convT'])])


    def forward(self, img, z,label=None):
        out = []
        gen_img = self.forward_generator_bn(z,self.g_ps,label)

        b1 = lbann.Identity(img)
        b2 = lbann.StopGradient(gen_img)
        b3 = lbann.Identity(gen_img)
        for bId in range(self.num_blocks):
            if label:
                out.append(self.forward_discriminator1(b1, bId,lbann.Identity(label)))
                out.append(self.forward_discriminator1(b2,bId,lbann.Identity(label)))
                if not self.alternate_updates:
                    out.append(self.forward_discriminator2(b3,bId,lbann.Identity(label)))
            else:
                out.append(self.forward_discriminator1(b1, bId))
                out.append(self.forward_discriminator1(b2,bId))
                if not self.alternate_updates:
                    out.append(self.forward_discriminator2(b3,bId))
        out.append(gen_img)   
        return out

    def forward_discriminator1(self,y,bId=0,x=None):
        print("Bid",bId)
        if x: y = self.imgX(y,x)
        if self.enable_subgraph: y = lbann.Identity(y, parallel_strategy = {'sub_branch_tag':bId+1,'enable_subgraph':True})
        x = self.d1_conv[bId][3](self.d1_conv[bId][2](self.d1_conv[bId][1](self.d1_conv[bId][0](y))))
        return self.d1_fc[bId](x)

    def forward_discriminator2(self,y,bId=0,x=None):
        if x: y = self.imgX(y,x)
        if self.enable_subgraph: y = lbann.Identity(y, parallel_strategy = {'sub_branch_tag':bId+1,'enable_subgraph':True})
        x = self.d2_conv[bId][3](self.d2_conv[bId][2](self.d2_conv[bId][1](self.d2_conv[bId][0](y))))
        return self.d2_fc[bId](x) 
 
    def forward_generator(self,z,ps=None,x=None):
        if x: z = self.noiseX(z,x)
        x = lbann.FullyConnected(z, num_neurons = np.prod(self.outc_dims), has_bias = True, device=self.g_device)
        x = lbann.Reshape(x, dims=self.outc_dims,name='gen_zin_reshape',device=self.g_device) 
        x = lbann.Relu(self.g_convT[0](x), name='g_relu0',parallel_strategy=ps,device=self.g_device)
        x = lbann.Relu(self.g_convT[1](x), name='g_relu1',parallel_strategy=ps,device=self.g_device)
        x = lbann.Relu(self.g_convT[2](x), name='g_relu2',parallel_strategy=ps,device=self.g_device)
        return self.g_convT3(x) 
 
    def forward_generator_bn(self,z,ps=None,x=None):
        if x: z = self.noiseX(z,x)
        x = lbann.FullyConnected(z, num_neurons = np.prod(self.outc_dims), has_bias = True, device=self.g_device)
        x = lbann.Reshape(x, dims=self.outc_dims,name='gen_zin_reshape',device=self.g_device) 
        x = lbann.Relu(lbann.BatchNormalization(self.g_convT[0](x),decay=0.9,scale_init=1.0,epsilon=1e-5))
        x = lbann.Relu(lbann.BatchNormalization(self.g_convT[1](x),decay=0.9,scale_init=1.0,epsilon=1e-5))
        x = lbann.Relu(lbann.BatchNormalization(self.g_convT[2](x),decay=0.9,scale_init=1.0,epsilon=1e-5))
        return self.g_convT3(x) 

    #Concatenate image (y) and label (x)
    def imgX(self, img, x):
        x_flat = lbann.Tessellate(x, dims=[self.input_width**3])
        img_flat = lbann.Reshape(img, dims=[self.input_channel*self.input_width**3])
        y_flat = lbann.Concatenation(img_flat, x_flat)
        y = lbann.Reshape(y_flat, dims=[self.input_channel+1] + 3*[self.input_width])
        return y
    #Concatenate noise(z) and label (x)
    #@todo: parameterize z dim
    def noiseX(self, z, x):
        x = lbann.Tessellate(x, dims=[64])
        x = lbann.Reshape(x, dims=[self.input_channel, 64])
        return lbann.Reshape(lbann.Concatenation(z,x,axis=0),dims=[self.input_channel+1,64])

