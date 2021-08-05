import lbann
import lbann.modules.base
import lbann.models.resnet
import lbann.modules as lm
from lbann.core.util import get_parallel_strategy_args
import numpy as np

def list2str(l):
    return ' '.join([str(i) for i in l])

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
        self.stride = stride
        self.bn_statistics_group_size = bn_statistics_group_size
        self.activation = activation
        self.use_bn = use_bn
        self.conv_weights = conv_weights
        self.ps = parallel_strategy

        # Initialize convolution
        self.conv = lbann.modules.Convolution3dModule(
            out_channels, kernel_size,
            stride=1, padding=padding,
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

        # Convolution
        layer = self.conv(x)

        # Batchnorm
        if self.use_bn:
            layer = lbann.BatchNormalization(
                layer, weights=self.bn_weights,
                statistics_group_size=self.bn_statistics_group_size,
                decay=0.999,
                parallel_strategy=self.ps,
                name='{0}_bn_instance{1}'.format(
                    self.name, self.instance))

        # Strided pooling
        # Note: Ideally we would do this immediately after the
        # convolution, but we run into issues since the tensor
        # overlaps don't match.
        ### @todo Support strided convolution in distconv
        if self.stride != 1:
            layer = lbann.Pooling(
                layer,
                num_dims=3,
                pool_dims_i=self.stride,
                pool_strides_i=self.stride,
                pool_mode='max',
                parallel_strategy=self.ps,
                name='{0}_pool_instance{1}'.format(
                self.name, self.instance)
            )

        # Activation
        if self.activation:
            layer = self.activation(
                layer,
                parallel_strategy=self.ps,
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
    """Generative model for 3D cosmology.

        Args:
            input_width (int): Size of input image i.e., one of 3D spatial dimensions (default=64)
            input_channel (int): Number of input channels (default=1).
            gen_device (str): Device allocation for generator network (default=GPU).
            disc_ps (lbann.ParallelStrategy): Parallel strategy for discriminator network (default=None).
            gen_ps (lbann.ParallelStrategy): Parallel strategy for generator network (default=None).
            use_bn (bool): Whether or not batch normalization layers are used (default= False).
            name (str): Module name.
    """

    global_count = 0  # Static counter, used for default names

    def __init__(self, input_width=64, input_channel=1,gen_device='GPU',
                 disc_ps=None, gen_ps=None,use_bn=False,name=None):

       self.instance = 0
       self.name = (name if name
                     else 'Exa3DGAN{0}'.format(Exa3DGAN.global_count))

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
       self.d1_conv = [convbnrelu(d_channels[i], kernel_size, stride, padding, self.use_bn, bn_stats_grp_sz, False,
                                  name=self.name+'_disc1_conv'+str(i),
                                  activation=lbann.Relu,
                                  parallel_strategy=self.d_ps,
                                  conv_weights=[lbann.Weights(initializer=self.inits['conv'])])
                   for i in range(len(d_channels))]
       self.d1_fc = fc(1,name=self.name+'_disc1_fc',
                       weights=[lbann.Weights(initializer=self.inits['dense'])])

       #stacked_discriminator, this will be frozen, no optimizer,
       #layer has to be named for callback
       self.d2_conv = [convbnrelu(d_channels[i], kernel_size, stride, padding, self.use_bn, bn_stats_grp_sz, False,
                                  name=self.name+'_disc2_conv'+str(i),
                                  activation=lbann.Relu,
                                  parallel_strategy=self.d_ps,
                                  conv_weights=[lbann.Weights(initializer=self.inits['conv'])])
                   for i in range(len(d_channels))]

       self.d2_fc = fc(1,name=self.name+'_disc2_fc',
                       weights=[lbann.Weights(initializer=self.inits['dense'])])


       g_channels = [256,128,64]
       kernel_size=2
       padding=0
       self.g_convT = [conv(g_channels[i], kernel_size, stride, padding, transpose=True,
                        name=self.name+'_gen'+str(i),
                        parallel_strategy=self.g_ps,
                       weights=[lbann.Weights(initializer=self.inits['convT'])])
                       for i in range(len(g_channels))]

       self.g_convT3 = conv(input_channel, kernel_size, stride, padding, activation=lbann.Tanh,
                            parallel_strategy=self.g_ps,
                            name='gen_img',transpose=True,
                            weights=[lbann.Weights(initializer=self.inits['convT'])])

    def forward(self, img, z):
        d1_real = self.forward_discriminator1(img)  #instance1
        gen_img = self.forward_generator(z,self.g_ps)
        d1_fake = self.forward_discriminator1(lbann.StopGradient(gen_img,name='stop_gradient')) #instance2
        d_adv = self.forward_discriminator2(gen_img) #instance 3 //need to freeze
        return d1_real,d1_fake,d_adv,gen_img

    def forward_discriminator1(self,y):
        x = self.d1_conv[3](self.d1_conv[2](self.d1_conv[1](self.d1_conv[0](y))))
        return self.d1_fc(x)

    def forward_discriminator2(self,y):
        x = self.d2_conv[3](self.d2_conv[2](self.d2_conv[1](self.d2_conv[0](y))))
        return self.d2_fc(x)

    def forward_generator(self,z,ps=None):
        x = lbann.FullyConnected(z, num_neurons = np.prod(self.outc_dims), has_bias = True, device=self.g_device)
        x = lbann.Reshape(x, dims=list2str(self.outc_dims),name='gen_zin_reshape',device=self.g_device)
        x = lbann.Relu(self.g_convT[0](x), name='g_relu0',parallel_strategy=ps,device=self.g_device)
        x = lbann.Relu(self.g_convT[1](x), name='g_relu1',parallel_strategy=ps,device=self.g_device)
        x = lbann.Relu(self.g_convT[2](x), name='g_relu2',parallel_strategy=ps,device=self.g_device)
        return self.g_convT3(x)
