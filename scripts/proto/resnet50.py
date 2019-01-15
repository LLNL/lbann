import lbann_proto as lp

blocks = [3, 4, 6, 3]  # Blocks for ResNet-50.
bn_stats_aggregation = 'global'

class ConvBNRelu2d:
    """Convolution -> Batch normalization -> ReLU"""

    def __init__(self, name, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=False, relu=True,
                 bn_stats_aggregation='global'):
        self.step = 0
        self.name = name
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        self.relu = relu
        self.bn_stats_aggregation = bn_stats_aggregation

    def __call__(self, x):
        name = (self.name
                if self.step
                else '{0}_step{1}'.format(self.name, self.step))
        self.step += 1
        conv = lp.Convolution(name + '_conv', x,
                              num_dims=2,
                              num_output_channels=self.out_channels,
                              has_vectors=False,
                              conv_dims_i=self.kernel_size,
                              conv_pads_i=self.padding,
                              conv_strides_i=self.stride,
                              conv_dilations_i=self.dilation,
                              has_bias=self.bias)
        bn = lp.BatchNormalization(name + '_bn', conv,
                                   decay=0.9, epsilon=1e-5,
                                   stats_aggregation=bn_stats_aggregation)
        if self.relu:
            return lp.Relu(name + '_relu', bn)
        else:
            return bn

class ResBottleneck:
    """ResNet bottleneck building block."""

    def __init__(self, name, mid_channels, out_channels, stride,
                 dilation=1, downsample=False,
                 bn_stats_aggregation='global'):
        self.step = 0
        self.name = name
        self.conv1 = ConvBNRelu2d(name + '_conv1', mid_channels, 1,
                                  stride=1, padding=0, dilation=1,
                                  bn_stats_aggregation=bn_stats_aggregation)
        self.conv2 = ConvBNRelu2d(name + '_conv2', mid_channels, 3,
                                  stride=stride, padding=dilation, dilation=dilation,
                                  bn_stats_aggregation=bn_stats_aggregation)
        self.conv3 = ConvBNRelu2d(name + '_conv3', out_channels, 1,
                                  stride=1, padding=0, dilation=1, relu=False,
                                  bn_stats_aggregation=bn_stats_aggregation)
        if downsample:
            self.downsample = ConvBNRelu2d(name + '_proj', out_channels, 1,
                                           stride=stride, padding=0,
                                           dilation=1, relu=False,
                                           bn_stats_aggregation=bn_stats_aggregation)
        else:
            self.downsample = None
        self.sum = lp.Sum(name + '_sum', data_layout)
        self.relu = lp.Relu(name + '_relu', data_layout)

    def __call__(self, x):
        name = (self.name
                if self.step
                else '{0}_step{1}'.format(self.name, self.step))
        self.step += 1
        conv = self.conv3(self.conv2(self.conv1(x)))
        if self.downsample is None:
            residual = x
        else:
            residual = self.downsample(x)
        sum = lp.Sum(name + '_sum', [conv, residual])
        return lp.Relu(name + '_relu', sum)

class ResBlock:
    """ResNet block, constructed of some number of bottleneck layers."""

    def __init__(self, name, num_layers, mid_channels, out_channels,
                 stride, dilation=1, bn_stats_aggregation='global'):
        self.layers = []
        self.layers.append(ResBottleneck(
            name + '_bottleneck1', mid_channels, out_channels,
            stride, dilation=dilation, downsample=True,
            bn_stats_aggregation=bn_stats_aggregation))
        for i in range(num_layers - 1):
            self.layers.append(ResBottleneck(
                name + '_bottleneck{0}'.format(i+2), mid_channels,
                out_channels, stride=1, dilation=dilation, downsample=False,
                bn_stats_aggregation=bn_stats_aggregation))

    def __call__(self, x):
        for l in self.layers:
            x = l(x)
        return x

# Construct layer graph.
input = lp.Input('input', io_buffer='partitioned')
images = lp.Split('images', input)
labels = lp.Split('labels', input)
conv1 = ConvBNRelu2d('conv1', 64, 7, stride=2, padding=3,
                     bn_stats_aggregation=bn_stats_aggregation)(images)
pool1 = lp.Pooling('pool1', conv1, num_dims=2, has_vectors=False,
                   pool_dims_i=3, pool_pads_i=1, pool_strides_i=2,
                   pool_mode='max')
block1 = ResBlock('block1', blocks[0], 64, 256, stride=1,
                  bn_stats_aggregation=bn_stats_aggregation)(pool1)
block2 = ResBlock('block2', blocks[1], 128, 512, stride=2,
                  bn_stats_aggregation=bn_stats_aggregation)(block1)
block3 = ResBlock('block3', blocks[2], 256, 1024, stride=2,
                  bn_stats_aggregation=bn_stats_aggregation)(block2)
block4 = ResBlock('block4', blocks[3], 512, 2048, stride=2,
                  bn_stats_aggregation=bn_stats_aggregation)(block3)
avgpool = lp.Pooling('avgpool', block4, num_dims=2, has_vectors=False,
                     pool_dims_i=7, pool_pads_i=0, pool_strides_i=1,
                     pool_mode='average')
fc = lp.FullyConnected('fc1000', avgpool, num_neurons=1000, has_bias=False)
softmax = lp.Softmax('prob', fc)
ce = lp.CrossEntropy('cross_entropy', [softmax, labels])
top1 = lp.CategoricalAccuracy('top1_accuracy', [softmax, labels])
top5 = lp.TopKCategoricalAccuracy('top5_accuracy', [softmax, labels], k=5)
layers = lp.traverse_layer_graph(input)

# Explicitly set up weights for all layers.
weights = []  # For saving the non-batchnorm weights.
def setup_weights(l):
    if type(l) == lp.Convolution:
        w = l.add_weights(lp.HeNormalInitializer(), name_suffix='_kernel')
        weights.append(w)
    elif type(l) == lp.BatchNormalization:
        # Set the initial scale of the last BN of each residual block to be 0.
        # A bit hackish, this assumes the particular naming scheme.
        if l.name.endswith('_conv3_bn'):
            l.add_weights(lp.ConstantInitializer(value=0.0), name_suffix='_scale')
        else:
            l.add_weights(lp.ConstantInitializer(value=1.0), name_suffix='_scale')
        l.add_weights(lp.ConstantInitializer(value=0.0), name_suffix='_bias')
    elif type(l) == lp.FullyConnected:
        w = l.add_weights(lp.NormalInitializer(mean=0.0, standard_deviation=0.01))
        weights.append(w)
map(setup_weights, layers)

# Objective function/metrics.
obj = lp.ObjectiveFunction(ce, [lp.L2WeightRegularization(
    scale_factor=1e-4, weights=weights)])
top1_metric = lp.Metric('categorical accuracy', top1, '%')
top5_metric = lp.Metric('top-5 categorical accuracy', top5, '%')

lp.save_model('resnet50.prototext', input, dl, 256, 90, obj,
              metrics=[top1_metric, top5_metric],
              callbacks=[lp.CallbackPrint(), lp.CallbackTimer(),
                         lp.CallbackDropFixedLearningRate(
                             drop_epoch=[30, 60, 80], amt=0.1)])
