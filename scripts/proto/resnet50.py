import lbann_proto as lp
import lbann_modules as lm

# ==============================================
# ResNet-specific modules
# ==============================================

class ConvBNRelu(lm.Module):
    """Convolution -> Batch normalization -> ReLU"""

    def __init__(self, out_channels, kernel_size,
                 stride=1, padding=0,
                 bn_zero_init=False, bn_stats_aggregation='local',
                 relu=True):
        super().__init__()
        self.conv = lm.Convolution2dModule(out_channels, kernel_size,
                                           stride=stride, padding=padding,
                                           bias=False)
        bn_scale = 0.0 if bn_zero_init else 1.0
        self.bn_weights = [lp.Weights(lp.ConstantInitializer(value=bn_scale)),
                           lp.Weights(lp.ConstantInitializer(value=0.0))]
        self.bn_stats_aggregation = bn_stats_aggregation
        self.relu = relu

    def __call__(self, x):
        conv = self.conv(x)
        bn = lp.BatchNormalization(conv, weights=self.bn_weights,
                                   stats_aggregation=self.bn_stats_aggregation)
        if self.relu:
            return lp.Relu(bn)
        else:
            return bn

class BottleneckBlock(lm.Module):

    def __init__(self, in_channels, mid_channels,
                 downsample=False, zero_init_residual=False,
                 bn_stats_aggregation='local'):
        super().__init__()
        self.out_channels = 4 * mid_channels
        self.conv1 = ConvBNRelu(mid_channels, 1,
                                stride=(2 if downsample else 1),
                                bn_stats_aggregation=bn_stats_aggregation)
        self.conv2 = ConvBNRelu(mid_channels, 3, padding=1,
                                bn_stats_aggregation=bn_stats_aggregation)
        self.conv3 = ConvBNRelu(self.out_channels, 1,
                                bn_zero_init=zero_init_residual,
                                bn_stats_aggregation=bn_stats_aggregation,
                                relu=False)
        if downsample:
            self.residual = ConvBNRelu(self.out_channels, 1,
                                       stride=2,
                                       bn_stats_aggregation=bn_stats_aggregation,
                                       relu=False)
        elif in_channels != self.out_channels:
            self.residual = ConvBNRelu(self.out_channels, 1,
                                       bn_stats_aggregation=bn_stats_aggregation,
                                       relu=False)
        else:
            self.residual = None

    def forward(self, x):
        y = self.conv3(self.conv2(self.conv1(x)))
        if self.residual is not None:
            x = self.residual(x)
        return lp.Relu(lp.Add([x, y]))

class ResNet(lm.Module):

    def __init__(self, block, output_size,
                 layer_sizes, layer_channels,
                 zero_init_residual=False,
                 bn_stats_aggregation='local'):
        super().__init__()
        self.conv1 = ConvBNRelu(layer_channels[0], 7,
                                stride=2, padding=3,
                                bn_stats_aggregation=bn_stats_aggregation)
        self.blocks = []
        for layer in range(len(layer_sizes)):
            mid_channels = layer_channels[layer]
            for i in range(layer_sizes[layer]):
                in_channels = (self.blocks[-1].out_channels
                               if self.blocks
                               else mid_channels)
                downsample = (i == 0 and layer > 0)
                self.blocks.append(block(in_channels, mid_channels,
                                         downsample=downsample,
                                         zero_init_residual=zero_init_residual,
                                         bn_stats_aggregation=bn_stats_aggregation))
        self.fc = lm.FullyConnectedModule(output_size, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = lp.Pooling(x, num_dims=2, has_vectors=False,
                       pool_dims_i=3, pool_pads_i=1, pool_strides_i=2,
                       pool_mode='max')
        for block in self.blocks:
            x = block(x)
        x = lp.ChannelwiseMean(x)
        return self.fc(x)

class ResNet50(ResNet):
    def __init__(self, output_size,
                 zero_init_residual=False,
                 bn_stats_aggregation='local'):
        super().__init__(BottleneckBlock, output_size,
                         (3,4,6,3), (64,128,256,512),
                         zero_init_residual=zero_init_residual,
                         bn_stats_aggregation=bn_stats_aggregation)

class ResNet101(ResNet):
    def __init__(self, output_size,
                 zero_init_residual=False,
                 bn_stats_aggregation='local'):
        super().__init__(BottleneckBlock, output_size,
                         (3,4,23,3), (64,128,256,512),
                         zero_init_residual=zero_init_residual,
                         bn_stats_aggregation=bn_stats_aggregation)

class ResNet152(ResNet):
    def __init__(self, output_size,
                 zero_init_residual=False,
                 bn_stats_aggregation='local'):
        super().__init__(BottleneckBlock, output_size,
                         (3,8,36,3), (64,128,256,512),
                         zero_init_residual=zero_init_residual,
                         bn_stats_aggregation=bn_stats_aggregation)

# ==============================================
# Construct model
# ==============================================

if __name__ == '__main__':

    # Options
    model_file = 'resnet50.prototext'
    output_size = 1000
    bn_stats_aggregation = 'local'
    zero_init_residual=False

    # Construct layer graph.
    input = lp.Input(io_buffer='partitioned')
    images = lp.Identity(input)
    labels = lp.Identity(input)
    resnet = ResNet50(output_size, zero_init_residual)
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

    # Export model to file
    lp.save_model(model_file, 256, 90,
                  layers=layers, objective_function=obj,
                  metrics=metrics, callbacks=callbacks)
