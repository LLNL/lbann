import lbann.lbann_proto as lp
import lbann.lbann_modules as lm

blocks = [3, 4, 6, 3]  # Blocks for ResNet-50.
bn_stats_aggregation = 'local'

class ConvBNRelu2d(lm.Module):
    """Convolution -> Batch normalization -> ReLU"""

    def __init__(self, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1,
                 bn_init_scale=1.0, bn_stats_aggregation='local',
                 relu=True):
        self.conv = lm.Convolution2dModule(out_channels, kernel_size,
                                           stride=stride, padding=padding,
                                           dilation=dilation, bias=False)
        self.bn_weights = [lp.Weights(lp.ConstantInitializer(value=bn_init_scale)),
                           lp.Weights(lp.ConstantInitializer(value=0.0))]
        self.bn_stats_aggregation = bn_stats_aggregation
        self.relu = relu

    def __call__(self, x):
        conv = self.conv(x)
        bn = lp.BatchNormalization(conv, weights=self.bn_weights,
                                   decay=0.9, epsilon=1e-5,
                                   stats_aggregation=self.bn_stats_aggregation)
        if self.relu:
            return lp.Relu(bn)
        else:
            return bn

class ResBottleneck(lm.Module):
    """ResNet bottleneck building block."""

    def __init__(self, mid_channels, out_channels,
                 stride, dilation=1, downsample=False,
                 bn_stats_aggregation='local'):
        self.conv1 = ConvBNRelu2d(mid_channels, 1,
                                  stride=1, padding=0, dilation=1,
                                  bn_stats_aggregation=bn_stats_aggregation)
        self.conv2 = ConvBNRelu2d(mid_channels, 3,
                                  stride=stride, padding=dilation, dilation=dilation,
                                  bn_stats_aggregation=bn_stats_aggregation)
        self.conv3 = ConvBNRelu2d(out_channels, 1,
                                  stride=1, padding=0, dilation=1,
                                  bn_init_scale=0.0,
                                  bn_stats_aggregation=bn_stats_aggregation,
                                  relu=False)
        if downsample:
            self.downsample = ConvBNRelu2d(out_channels, 1,
                                           stride=stride, padding=0, dilation=1,
                                           bn_stats_aggregation=bn_stats_aggregation,
                                           relu=False)
        else:
            self.downsample = None

    def __call__(self, x):
        conv = self.conv3(self.conv2(self.conv1(x)))
        if self.downsample is None:
            residual = x
        else:
            residual = self.downsample(x)
        return lp.Relu(lp.Add([conv, residual]))

class ResBlock:
    """ResNet block, constructed of some number of bottleneck layers.

    Here we use "layer" in the PyTorch/TensorFlow/Keras sense of the
    word, not the way LBANN uses it.

    """

    def __init__(self, num_layers, mid_channels, out_channels,
                 stride, dilation=1, bn_stats_aggregation='local'):
        self.layers = []
        self.layers.append(ResBottleneck(
            mid_channels, out_channels,
            stride, dilation=dilation, downsample=True,
            bn_stats_aggregation=bn_stats_aggregation))
        for i in range(num_layers - 1):
            self.layers.append(ResBottleneck(
                mid_channels, out_channels,
                stride=1, dilation=dilation, downsample=False,
                bn_stats_aggregation=bn_stats_aggregation))

    def __call__(self, x):
        for l in self.layers:
            x = l(x)
        return x

# Construct layer graph.
input = lp.Input(io_buffer='partitioned')
images = lp.Identity(input)
labels = lp.Identity(input)
conv1 = ConvBNRelu2d(64, 7, stride=2, padding=3,
                     bn_stats_aggregation=bn_stats_aggregation)(images)
pool1 = lp.Pooling(conv1, num_dims=2, has_vectors=False,
                   pool_dims_i=3, pool_pads_i=1, pool_strides_i=2,
                   pool_mode='max')
block1 = ResBlock(blocks[0], 64, 256, stride=1,
                  bn_stats_aggregation=bn_stats_aggregation)(pool1)
block2 = ResBlock(blocks[1], 128, 512, stride=2,
                  bn_stats_aggregation=bn_stats_aggregation)(block1)
block3 = ResBlock(blocks[2], 256, 1024, stride=2,
                  bn_stats_aggregation=bn_stats_aggregation)(block2)
block4 = ResBlock(blocks[3], 512, 2048, stride=2,
                  bn_stats_aggregation=bn_stats_aggregation)(block3)
avgpool = lp.Pooling(block4, num_dims=2, has_vectors=False,
                     pool_dims_i=7, pool_pads_i=0, pool_strides_i=1,
                     pool_mode='average')
fc = lp.FullyConnected(avgpool,
                       weights=lp.Weights(initializer=lp.HeNormalInitializer()),
                       hint_layer=labels, has_bias=False)
softmax = lp.Softmax(fc)
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

# Export model to file
lp.save_model('resnet50.prototext', 256, 90,
              layers=layers, objective_function=obj,
              metrics=metrics, callbacks=callbacks)
