import lbann
import lbann.modules

# ==============================================
# Helper modules
# ==============================================

class ConvBNRelu(lbann.modules.Module):
    """Convolution -> Batch normalization -> ReLU

    Basic unit for ResNets. Assumes image data in NCHW format.

    """

    def __init__(self, out_channels, kernel_size, stride, padding,
                 bn_zero_init, bn_statistics_group_size,
                 relu, name):
        """Initialize ConvBNRelu module.

        Args:
            out_channels (int): Number of output channels, i.e. number
                of convolution filters.
            kernel_size (int): Size of convolution kernel.
            stride (int): Convolution stride.
            padding (int): Convolution padding.
            bn_zero_init (bool): Zero-initialize batch normalization
                scale.
            bn_statistics_group_size (int): Group size for aggregating
                batch normalization statistics.
            relu (bool): Apply ReLU activation.
            name (str): Module name.

        """
        super().__init__()
        self.name = name
        self.instance = 0

        # Initialize convolution
        self.conv = lbann.modules.Convolution2dModule(
            out_channels, kernel_size,
            stride=stride, padding=padding,
            bias=False,
            name=self.name + '_conv')

        # Initialize batch normalization
        bn_scale_init = 0.0 if bn_zero_init else 1.0
        bn_scale = lbann.Weights(
            initializer=lbann.ConstantInitializer(value=bn_scale_init),
            name=self.name + '_bn_scale')
        bn_bias = lbann.Weights(
            initializer=lbann.ConstantInitializer(value=0.0),
            name=self.name + '_bn_bias')
        self.bn_weights = [bn_scale, bn_bias]
        self.bn_statistics_group_size = bn_statistics_group_size

        # Initialize ReLU
        self.relu = relu

    def forward(self, x):
        self.instance += 1
        conv = self.conv(x)
        bn = lbann.BatchNormalization(
            conv, weights=self.bn_weights,
            statistics_group_size=(-1 if self.bn_statistics_group_size == 0
                                   else self.bn_statistics_group_size),
            name='{0}_bn_instance{1}'.format(self.name,self.instance))
        if self.relu:
            return lbann.Relu(
                bn, name='{0}_relu_instance{1}'.format(self.name,self.instance))
        else:
            return bn

class BasicBlock(lbann.modules.Module):
    """Residual block without bottlenecking.

    The number of output channels is the same as the number of
    internal channels. Assumes image data in NCHW format. This is the
    residual block used in ResNet-{18,34}.

    """

    def __init__(self, in_channels, mid_channels,
                 downsample, zero_init_residual,
                 bn_statistics_group_size, name, width=1):
        """Initialize residual block.

        Args:
            in_channels (int): Number of input channels.
            mid_channels (int): Number of channels in residual branch.
            downsample (bool): Perform spatial downsampling (by a
                factor of 2 in each spatial dimension).
            zero_init_residual (bool): Zero-initialize the scale in
                the final batch normalization in the residual branch.
            bn_statistics_group_size (int): Group size for aggregating
                batch normalization statistics.
            name (str): Module name.
            width (float, optional): Width growth factor for 3x3
                convolutions.

        """
        super().__init__()
        self.name = name
        self.instance = 0
        mid_channels = int(mid_channels * width)
        self.out_channels = mid_channels

        # Skip connection
        if downsample:
            self.branch1 = ConvBNRelu(self.out_channels, 1, 2, 0,
                                      False, bn_statistics_group_size,
                                      False, self.name + '_branch1')
        elif in_channels != self.out_channels:
            self.branch1 = ConvBNRelu(self.out_channels, 1, 1, 0,
                                      False, bn_statistics_group_size,
                                      False, self.name + '_branch1')
        else:
            self.branch1 = None

        # Residual branch
        self.branch2a = ConvBNRelu(mid_channels, 3,
                                   (2 if downsample else 1), 1,
                                   False, bn_statistics_group_size,
                                   True, self.name + '_branch2a')
        self.branch2b = ConvBNRelu(self.out_channels, 3, 1, 1,
                                   zero_init_residual,
                                   bn_statistics_group_size,
                                   False, self.name + '_branch2b')

    def forward(self, x):
        self.instance += 1
        y1 = self.branch1(x) if self.branch1 else x
        y2 = self.branch2b(self.branch2a(x))
        z = lbann.Add([y1, y2],
                      name='{0}_sum_instance{1}'.format(self.name,self.instance))
        return lbann.Relu(z, name='{0}_relu_instance{1}'.format(self.name,self.instance))

class BottleneckBlock(lbann.modules.Module):
    """Residual block with bottlenecking.

    The number of output channels is four times the number of internal
    channels. Assumes image data in NCHW format. This is the residual
    block used in ResNet-{50,101,152}.

    """

    def __init__(self, in_channels, mid_channels,
                 downsample, zero_init_residual,
                 bn_statistics_group_size, name, width=1):
        """Initialize residual block.

        Args:
            in_channels (int): Number of input channels.
            mid_channels (int): Number of channels in residual branch.
            downsample (bool): Perform spatial downsampling (by a
                factor of 2 in each spatial dimension).
            zero_init_residual (bool): Zero-initialize the scale in
                the final batch normalization in the residual branch.
            bn_statistics_group_size (int): Group size for aggregating
                batch normalization statistics.
            name (str): Module name.
            width (float, optional): Width growth factor for 3x3
                convolutions.

        """
        super().__init__()
        self.name = name
        self.instance = 0
        self.out_channels = 4 * mid_channels
        # Width factor does not grow the output channel size.
        mid_channels = int(mid_channels * width)

        # Skip connection
        if downsample:
            self.branch1 = ConvBNRelu(self.out_channels, 1, 2, 0,
                                      False, bn_statistics_group_size,
                                      False, self.name + '_branch1')
        elif in_channels != self.out_channels:
            self.branch1 = ConvBNRelu(self.out_channels, 1, 1, 0,
                                      False, bn_statistics_group_size,
                                      False, self.name + '_branch1')
        else:
            self.branch1 = None

        # Residual branch
        self.branch2a = ConvBNRelu(mid_channels, 1, 1, 0,
                                   False, bn_statistics_group_size,
                                   True, self.name + '_branch2a')
        self.branch2b = ConvBNRelu(mid_channels, 3,
                                   (2 if downsample else 1), 1,
                                   False, bn_statistics_group_size,
                                   True, self.name + '_branch2b')
        self.branch2c = ConvBNRelu(self.out_channels, 1, 1, 0,
                                   zero_init_residual,
                                   bn_statistics_group_size,
                                   False, self.name + '_branch2c')

    def forward(self, x):
        self.instance += 1
        y1 = self.branch1(x) if self.branch1 else x
        y2 = self.branch2c(self.branch2b(self.branch2a(x)))
        z = lbann.Add([y1, y2],
                      name='{0}_sum_instance{1}'.format(self.name,self.instance))
        return lbann.Relu(z, name='{0}_relu_instance{1}'.format(self.name,self.instance))

# ==============================================
# ResNet modules
# ==============================================

class ResNet(lbann.modules.Module):
    """Residual neural network.

    A ResNet is comprised of residual blocks, which are small
    convolutional networks with a skip connection. These blocks are
    grouped into "layers" (this is a horribly overloaded term, but we
    are following the common usage). At the first block in each layer
    (except the first), the spatial dimensions are all downsampled by
    a factor of 2. A fully-connected layer is applied at the end to
    obtain an output tensor of the desired dimension. Input data is
    assumed to be image data in NCHW format.

    See:
        Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. "Deep
        residual learning for image recognition." In Proceedings of
        the IEEE Conference on Computer Vision and Pattern
        Recognition, pp. 770-778. 2016.

    """


    def __init__(self, block, output_size,
                 layer_sizes, layer_channels,
                 zero_init_residual, bn_statistics_group_size,
                 name, width=1):
        """Initialize ResNet.

        Args:
            block (type): Residual block type, which should be a
                `lbann.modules.Module`.
            output_size (int): Size of output tensor.
            layer_sizes (`Iterable` containing `int`s): Number of
                blocks in each ResNet layer.
            layer_channels (`Iterable` containing `int`s): Number of
                internal channels in each ResNet layer.
            zero_init_residual (bool): Whether to initialize the final
                batch normalization in residual branches with zeros.
            bn_statistics_group_size (int): Group size for aggregating
                batch normalization statistics.
            name (str): Module name.
            width (float, optional): Width growth factor.

        """
        super().__init__()
        self.name = name
        self.instance = 0
        self.conv1 = ConvBNRelu(layer_channels[0], 7, 2, 3,
                                False, bn_statistics_group_size,
                                True, self.name + '_conv1')
        self.blocks = []
        for layer in range(len(layer_sizes)):
            mid_channels = layer_channels[layer]
            for i in range(layer_sizes[layer]):
                in_channels = (self.blocks[-1].out_channels
                               if self.blocks
                               else mid_channels)
                downsample = (i == 0 and layer > 0)
                b = block(in_channels, mid_channels,
                          downsample, zero_init_residual,
                          bn_statistics_group_size,
                          '{0}_layer{1}_block{2}'.format(self.name, layer, i),
                          width=width)
                self.blocks.append(b)
        self.fc = lbann.modules.FullyConnectedModule(
            output_size, bias=False, name=self.name + '_fc')

    def forward(self, x):
        self.instance += 1
        x = self.conv1(x)
        x = lbann.Pooling(x, num_dims=2, has_vectors=False,
                          pool_dims_i=3, pool_pads_i=1, pool_strides_i=2,
                          pool_mode='max',
                          name='{0}_pool1_instance{1}'.format(self.name,self.instance))
        for b in self.blocks:
            x = b(x)
        x = lbann.ChannelwiseMean(x, name='{0}_avgpool_instance{1}'.format(self.name,self.instance))
        return self.fc(x)

class ResNet18(ResNet):
    """ResNet-18 neural network.

    Assumes image data in NCHW format.

    See:
        Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. "Deep
        residual learning for image recognition." In Proceedings of
        the IEEE Conference on Computer Vision and Pattern
        Recognition, pp. 770-778. 2016.

    """

    global_count = 0  # Static counter, used for default names

    def __init__(self, output_size,
                 zero_init_residual=True,
                 bn_statistics_group_size=1,
                 name=None, width=1):
        """Initialize ResNet-18.

        Args:
            output_size (int): Size of output tensor.
            zero_init_residual (bool, optional): Whether to initialize
                the final batch normalization in residual branches
                with zeros.
            bn_statistics_group_size (str, optional): Group size for
                aggregating batch normalization statistics.
            name (str, optional): Module name
                (default: 'resnet18_module<index>')
            width (float, optional): Width growth factor.

        """
        ResNet18.global_count += 1
        if name is None:
            name = 'resnet18_module{0}'.format(ResNet18.global_count)
        super().__init__(BasicBlock, output_size,
                         (2,2,2,2), (64,128,256,512),
                         zero_init_residual, bn_statistics_group_size,
                         name, width=width)

class ResNet34(ResNet):
    """ResNet-34 neural network.

    Assumes image data in NCHW format.

    See:
        Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. "Deep
        residual learning for image recognition." In Proceedings of
        the IEEE Conference on Computer Vision and Pattern
        Recognition, pp. 770-778. 2016.

    """

    global_count = 0  # Static counter, used for default names

    def __init__(self, output_size,
                 zero_init_residual=True,
                 bn_statistics_group_size=1,
                 name=None, width=1):
        """Initialize ResNet-34.

        Args:
            output_size (int): Size of output tensor.
            zero_init_residual (bool, optional): Whether to initialize
                the final batch normalization in residual branches
                with zeros.
            bn_statistics_group_size (str, optional): Group size for
                aggregating batch normalization statistics.
            name (str, optional): Module name
                (default: 'resnet34_module<index>')
            width (float, optional): Width growth factor.

        """
        ResNet34.global_count += 1
        if name is None:
            name = 'resnet34_module{0}'.format(ResNet34.global_count)
        super().__init__(BasicBlock, output_size,
                         (3,4,6,3), (64,128,256,512),
                         zero_init_residual, bn_statistics_group_size,
                         name, width=width)

class ResNet50(ResNet):
    """ResNet-50 neural network.

    Assumes image data in NCHW format.

    See:
        Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. "Deep
        residual learning for image recognition." In Proceedings of
        the IEEE Conference on Computer Vision and Pattern
        Recognition, pp. 770-778. 2016.

    """

    global_count = 0  # Static counter, used for default names

    def __init__(self, output_size,
                 zero_init_residual=True,
                 bn_statistics_group_size=1,
                 name=None, width=1):
        """Initialize ResNet-50.

        Args:
            output_size (int): Size of output tensor.
            zero_init_residual (bool, optional): Whether to initialize
                the final batch normalization in residual branches
                with zeros.
            bn_statistics_group_size (str, optional): Group size for
                aggregating batch normalization statistics.
            name (str, optional): Module name
                (default: 'resnet50_module<index>')
            width (float, optional): Width growth factor.

        """
        ResNet50.global_count += 1
        if name is None:
            name = 'resnet50_module{0}'.format(ResNet50.global_count)
        super().__init__(BottleneckBlock, output_size,
                         (3,4,6,3), (64,128,256,512),
                         zero_init_residual, bn_statistics_group_size,
                         name, width=width)

class ResNet101(ResNet):
    """ResNet-101 neural network.

    Assumes image data in NCHW format.

    See:
        Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. "Deep
        residual learning for image recognition." In Proceedings of
        the IEEE Conference on Computer Vision and Pattern
        Recognition, pp. 770-778. 2016.

    """

    global_count = 0  # Static counter, used for default names

    def __init__(self, output_size,
                 zero_init_residual=True,
                 bn_statistics_group_size=1,
                 name=None, width=1):
        """Initialize ResNet-101.

        Args:
            output_size (int): Size of output tensor.
            zero_init_residual (bool, optional): Whether to initialize
                the final batch normalization in residual branches
                with zeros.
            bn_statistics_group_size (str, optional): Group size for
                aggregating batch normalization statistics.
            name (str, optional): Module name
                (default: 'resnet101_module<index>')
            width (float, optional): Width growth factor.

        """
        ResNet101.global_count += 1
        if name is None:
            name = 'resnet101_module{0}'.format(ResNet101.global_count)
        super().__init__(BottleneckBlock, output_size,
                         (3,4,23,3), (64,128,256,512),
                         zero_init_residual, bn_statistics_group_size,
                         name, width=width)

class ResNet152(ResNet):
    """ResNet-152 neural network.

    Assumes image data in NCHW format.

    See:
        Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. "Deep
        residual learning for image recognition." In Proceedings of
        the IEEE Conference on Computer Vision and Pattern
        Recognition, pp. 770-778. 2016.

    """

    global_count = 0  # Static counter, used for default names

    def __init__(self, output_size,
                 zero_init_residual=True,
                 bn_statistics_group_size=1,
                 name=None, width=1):
        """Initialize ResNet-152.

        Args:
            output_size (int): Size of output tensor.
            zero_init_residual (bool, optional): Whether to initialize
                the final batch normalization in residual branches
                with zeros.
            bn_statistics_group_size (str, optional): Group size for
                aggregating batch normalization statistics.
            name (str, optional): Module name
                (default: 'resnet152_module<index>')
            width (float, optional): Width growth factor.

        """
        ResNet152.global_count += 1
        if name is None:
            name = 'resnet152_module{0}'.format(ResNet152.global_count)
        super().__init__(BottleneckBlock, output_size,
                         (3,8,36,3), (64,128,256,512),
                         zero_init_residual, bn_statistics_group_size,
                         name, width=width)
