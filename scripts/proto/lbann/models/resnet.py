import lbann.proto as lp
import lbann.modules as lm

# ==============================================
# Helper modules
# ==============================================

class ConvBNRelu(lm.Module):
    """Convolution -> Batch normalization -> ReLU

    Basic unit for ResNets. Assumes image data in NCHW format.

    """

    def __init__(self, out_channels, kernel_size, stride, padding,
                 bn_zero_init, bn_stats_aggregation,
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
            bn_stats_aggregation (str): Aggregation mode for batch
                normalization statistics.
            relu (bool): Apply ReLU activation.
            name (str): Module name.

        """
        super().__init__()
        self.name = name
        self.instance = 0

        # Initialize convolution
        self.conv = lm.Convolution2dModule(out_channels, kernel_size,
                                           stride=stride, padding=padding,
                                           bias=False,
                                           name=self.name + '_conv')

        # Initialize batch normalization
        bn_scale_init = 0.0 if bn_zero_init else 1.0
        bn_scale = lp.Weights(initializer=lp.ConstantInitializer(value=bn_scale_init),
                              name=self.name + '_bn_scale')
        bn_bias = lp.Weights(initializer=lp.ConstantInitializer(value=0.0),
                             name=self.name + '_bn_bias')
        self.bn_weights = [bn_scale, bn_bias]
        self.bn_stats_aggregation = bn_stats_aggregation

        # Initialize ReLU
        self.relu = relu

    def forward(self, x):
        self.instance += 1
        conv = self.conv(x)
        bn = lp.BatchNormalization(conv, weights=self.bn_weights,
                                   stats_aggregation=self.bn_stats_aggregation,
                                   name='{0}_bn_instance{1}'.format(self.name,self.instance))
        if self.relu:
            return lp.Relu(bn, name='{0}_relu_instance{1}'.format(self.name,self.instance))
        else:
            return bn

class BasicBlock(lm.Module):
    """Residual block without bottlenecking.

    The number of output channels is the same as the number of
    internal channels. Assumes image data in NCHW format. This is the
    residual block used in ResNet-{18,34}.

    """

    def __init__(self, in_channels, mid_channels,
                 downsample, zero_init_residual,
                 bn_stats_aggregation, name):
        """Initialize residual block.

        Args:
            in_channels (int): Number of input channels.
            mid_channels (int): Number of channels in residual branch.
            downsample (bool): Perform spatial downsampling (by a
                factor of 2 in each spatial dimension).
            zero_init_residual (bool): Zero-initialize the scale in
                the final batch normalization in the residual branch.
            bn_stats_aggregation (str): Aggregation mode for batch
                normalization statistics.
            name (str): Module name.

        """
        super().__init__()
        self.name = name
        self.instance = 0
        self.out_channels = mid_channels

        # Skip connection
        if downsample:
            self.branch1 = ConvBNRelu(self.out_channels, 1, 2, 0,
                                      False, bn_stats_aggregation,
                                      False, self.name + '_branch1')
        elif in_channels != self.out_channels:
            self.branch1 = ConvBNRelu(self.out_channels, 1, 1, 0,
                                      False, bn_stats_aggregation,
                                      False, self.name + '_branch1')
        else:
            self.branch1 = None

        # Residual branch
        self.branch2a = ConvBNRelu(mid_channels, 3,
                                   (2 if downsample else 1), 1,
                                   False, bn_stats_aggregation,
                                   True, self.name + '_branch2a')
        self.branch2b = ConvBNRelu(self.out_channels, 3, 1, 1,
                                   zero_init_residual,
                                   bn_stats_aggregation,
                                   False, self.name + '_branch2b')

    def forward(self, x):
        self.instance += 1
        y1 = self.branch1(x) if self.branch1 else x
        y2 = self.branch2b(self.branch2a(x))
        z = lp.Add([y1, y2],
                   name='{0}_sum_instance{1}'.format(self.name,self.instance))
        return lp.Relu(z, name='{0}_relu_instance{1}'.format(self.name,self.instance))

class BottleneckBlock(lm.Module):
    """Residual block with bottlenecking.

    The number of output channels is four times the number of internal
    channels. Assumes image data in NCHW format. This is the residual
    block used in ResNet-{50,101,152}.

    """

    def __init__(self, in_channels, mid_channels,
                 downsample, zero_init_residual,
                 bn_stats_aggregation, name):
        """Initialize residual block.

        Args:
            in_channels (int): Number of input channels.
            mid_channels (int): Number of channels in residual branch.
            downsample (bool): Perform spatial downsampling (by a
                factor of 2 in each spatial dimension).
            zero_init_residual (bool): Zero-initialize the scale in
                the final batch normalization in the residual branch.
            bn_stats_aggregation (str): Aggregation mode for batch
                normalization statistics.
            name (str): Module name.

        """
        super().__init__()
        self.name = name
        self.instance = 0
        self.out_channels = 4 * mid_channels

        # Skip connection
        if downsample:
            self.branch1 = ConvBNRelu(self.out_channels, 1, 2, 0,
                                      False, bn_stats_aggregation,
                                      False, self.name + '_branch1')
        elif in_channels != self.out_channels:
            self.branch1 = ConvBNRelu(self.out_channels, 1, 1, 0,
                                      False, bn_stats_aggregation,
                                      False, self.name + '_branch1')
        else:
            self.branch1 = None

        # Residual branch
        self.branch2a = ConvBNRelu(mid_channels, 1, 1, 0,
                                   False, bn_stats_aggregation,
                                   True, self.name + '_branch2a')
        self.branch2b = ConvBNRelu(mid_channels, 3,
                                   (2 if downsample else 1), 1,
                                   False, bn_stats_aggregation,
                                   True, self.name + '_branch2b')
        self.branch2c = ConvBNRelu(self.out_channels, 1, 1, 0,
                                   zero_init_residual,
                                   bn_stats_aggregation,
                                   False, self.name + '_branch2c')

    def forward(self, x):
        self.instance += 1
        y1 = self.branch1(x) if self.branch1 else x
        y2 = self.branch2c(self.branch2b(self.branch2a(x)))
        z = lp.Add([y1, y2],
                   name='{0}_sum_instance{1}'.format(self.name,self.instance))
        return lp.Relu(z, name='{0}_relu_instance{1}'.format(self.name,self.instance))

# ==============================================
# ResNet modules
# ==============================================

class ResNet(lm.Module):
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
                 zero_init_residual, bn_stats_aggregation,
                 name):
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
            bn_stats_aggregation (str): Aggregation mode for batch
                normalization statistics.
            name (str): Module name.

        """
        super().__init__()
        self.name = name
        self.instance = 0
        self.conv1 = ConvBNRelu(layer_channels[0], 7, 2, 3,
                                False, bn_stats_aggregation,
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
                          bn_stats_aggregation,
                          '{0}_layer{1}_block{2}'.format(self.name, layer, i))
                self.blocks.append(b)
        self.fc = lm.FullyConnectedModule(output_size, bias=False,
                                          name=self.name + '_fc')

    def forward(self, x):
        self.instance += 1
        x = self.conv1(x)
        x = lp.Pooling(x, num_dims=2, has_vectors=False,
                       pool_dims_i=3, pool_pads_i=1, pool_strides_i=2,
                       pool_mode='max',
                       name='{0}_pool1_instance{1}'.format(self.name,self.instance))
        for b in self.blocks:
            x = b(x)
        x = lp.ChannelwiseMean(x, name='{0}_avgpool_instance{1}'.format(self.name,self.instance))
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
                 bn_stats_aggregation='local',
                 name=None):
        """Initialize ResNet-18.

        Args:
            output_size (int): Size of output tensor.
            zero_init_residual (bool, optional): Whether to initialize
                the final batch normalization in residual branches
                with zeros.
            bn_stats_aggregation (str, optional): Aggregation mode for
                batch normalization statistics.
            name (str, optional): Module name
                (default: 'resnet18_module<index>')

        """
        ResNet18.global_count += 1
        if name is None:
            name = 'resnet18_module{0}'.format(ResNet18.global_count)
        super().__init__(BasicBlock, output_size,
                         (2,2,2,2), (64,128,256,512),
                         zero_init_residual, bn_stats_aggregation,
                         name)

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
                 bn_stats_aggregation='local',
                 name=None):
        """Initialize ResNet-34.

        Args:
            output_size (int): Size of output tensor.
            zero_init_residual (bool, optional): Whether to initialize
                the final batch normalization in residual branches
                with zeros.
            bn_stats_aggregation (str, optional): Aggregation mode for
                batch normalization statistics.
            name (str, optional): Module name
                (default: 'resnet34_module<index>')

        """
        ResNet34.global_count += 1
        if name is None:
            name = 'resnet34_module{0}'.format(ResNet34.global_count)
        super().__init__(BasicBlock, output_size,
                         (3,4,6,3), (64,128,256,512),
                         zero_init_residual, bn_stats_aggregation,
                         name)

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
                 bn_stats_aggregation='local',
                 name=None):
        """Initialize ResNet-50.

        Args:
            output_size (int): Size of output tensor.
            zero_init_residual (bool, optional): Whether to initialize
                the final batch normalization in residual branches
                with zeros.
            bn_stats_aggregation (str, optional): Aggregation mode for
                batch normalization statistics.
            name (str, optional): Module name
                (default: 'resnet50_module<index>')

        """
        ResNet50.global_count += 1
        if name is None:
            name = 'resnet50_module{0}'.format(ResNet50.global_count)
        super().__init__(BottleneckBlock, output_size,
                         (3,4,6,3), (64,128,256,512),
                         zero_init_residual, bn_stats_aggregation,
                         name)

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
                 bn_stats_aggregation='local',
                 name=None):
        """Initialize ResNet-101.

        Args:
            output_size (int): Size of output tensor.
            zero_init_residual (bool, optional): Whether to initialize
                the final batch normalization in residual branches
                with zeros.
            bn_stats_aggregation (str, optional): Aggregation mode for
                batch normalization statistics.
            name (str, optional): Module name
                (default: 'resnet101_module<index>')

        """
        ResNet101.global_count += 1
        if name is None:
            name = 'resnet101_module{0}'.format(ResNet101.global_count)
        super().__init__(BottleneckBlock, output_size,
                         (3,4,23,3), (64,128,256,512),
                         zero_init_residual, bn_stats_aggregation,
                         name)

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
                 bn_stats_aggregation='local',
                 name=None):
        """Initialize ResNet-152.

        Args:
            output_size (int): Size of output tensor.
            zero_init_residual (bool, optional): Whether to initialize
                the final batch normalization in residual branches
                with zeros.
            bn_stats_aggregation (str, optional): Aggregation mode for
                batch normalization statistics.
            name (str, optional): Module name
                (default: 'resnet152_module<index>')

        """
        ResNet152.global_count += 1
        if name is None:
            name = 'resnet152_module{0}'.format(ResNet152.global_count)
        super().__init__(BottleneckBlock, output_size,
                         (3,8,36,3), (64,128,256,512),
                         zero_init_residual, bn_stats_aggregation,
                         name)

# ==============================================
# Export prototext
# ==============================================

if __name__ == '__main__':

    # Options
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'file',
        nargs='?', default='model.prototext', type=str,
        help='exported prototext file')
    parser.add_argument(
        '--resnet',
        action='store', default=50, type=int,
        choices=(18, 34, 50, 101, 152),
        help='ResNet variant (default: 50)')
    parser.add_argument(
        '--num-labels', action='store', default=1000, type=int,
        help='number of data classes (default: 1000)')
    parser.add_argument(
        '--bn-stats-aggregation',
        action='store', default='local', type=str,
        help=('aggregation mode for batch normalization statistics '
              '(default: "local")'))
    parser.add_argument(
        '--mbsize', action='store', default=256, type=int,
        help='mini-batch size (default: 256)')
    parser.add_argument(
        '--epochs', action='store', default=90, type=int,
        help='number of epochs (default: 90)')
    parser.add_argument(
        '--warmup', action='store_true',
        help='Use a linear warmup (default: false)')
    args = parser.parse_args()

    # Choose ResNet variant
    resnet_variant_dict = {18: ResNet18, 34: ResNet34,
                           50: ResNet50, 101: ResNet101, 152: ResNet152}
    resnet = resnet_variant_dict[args.resnet](
        args.num_labels,
        bn_stats_aggregation=args.bn_stats_aggregation)

    # Construct layer graph.
    input = lp.Input()
    images = lp.Identity(input)
    labels = lp.Identity(input)
    softmax = lp.Softmax(resnet(images))
    ce = lp.CrossEntropy([softmax, labels])
    top1 = lp.CategoricalAccuracy([softmax, labels])
    top5 = lp.TopKCategoricalAccuracy([softmax, labels], k=5)
    layers = list(lp.traverse_layer_graph(input))

    # Setup objective function
    l2_reg_weights = set()
    for l in layers:
        if type(l) == lp.Convolution or type(l) == lp.FullyConnected:
            l2_reg_weights.update(l.weights)
    l2_reg = lp.L2WeightRegularization(weights=l2_reg_weights, scale=1e-4)
    obj = lp.ObjectiveFunction([ce, l2_reg])

    # Set up metrics and callbacks
    metrics = [lp.Metric(top1, name='categorical accuracy', unit='%'),
               lp.Metric(top5, name='top-5 categorical accuracy', unit='%')]
    callbacks = [lp.CallbackPrint(),
                 lp.CallbackTimer(),
                 lp.CallbackDropFixedLearningRate(
                     drop_epoch=[30, 60, 80], amt=0.1)]
    if args.warmup:
        callbacks.append(lp.CallbackLinearGrowthLearningRate(
            target=0.1*args.mbsize / 256, num_epochs=5))

    # Export model to file
    model = lp.Model(args.mbsize, args.epochs,
                     layers=layers, objective_function=obj,
                     metrics=metrics, callbacks=callbacks)
    model.save_proto(args.file)
