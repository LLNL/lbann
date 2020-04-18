import lbann
import lbann.modules
import resnet

class BatchNormModule(lbann.modules.Module):

    global_count = 0    # Static counter, used for default names

    def __init__(self,
                 statistics_group_size=1,
                 name=None,
                 data_layout='data_parallel'):
        super().__init__()
        BatchNormModule.global_count += 1
        self.instance = 0
        self.statistics_group_size = statistics_group_size
        self.name = (name
                     if name
                     else 'bnmodule{0}'.format(BatchNormModule.global_count))
        self.data_layout = data_layout

        # Initialize weights
        self.scale = lbann.Weights(
            initializer=lbann.ConstantInitializer(value=1.0),
            name=self.name + '_scale')
        self.bias = lbann.Weights(
            initializer=lbann.ConstantInitializer(value=0.0),
            name=self.name + '_bias')
        self.running_mean = lbann.Weights(
            initializer=lbann.ConstantInitializer(value=0.0),
            name=self.name + '_running_mean')
        self.running_variance = lbann.Weights(
            initializer=lbann.ConstantInitializer(value=1.0),
            name=self.name + '_running_variance')

    def forward(self, x):
        self.instance += 1
        name = '{0}_instance{1}'.format(self.name, self.instance)
        return lbann.BatchNormalization(
            x,
            weights=[self.scale, self.bias,
                     self.running_mean, self.running_variance],
            decay=0.9,
            scale_init=1.0,
            bias_init=0.0,
            epsilon=1e-5,
            statistics_group_size=self.statistics_group_size,
            name=name,
            data_layout=self.data_layout)

class ConvBnRelu(lbann.modules.Module):

    global_count = 0    # Static counter, used for default names

    def __init__(self,
                 out_channels, kernel_size,
                 stride=1, padding=0,
                 statistics_group_size=1,
                 name=None):
        super().__init__()
        ConvBnRelu.global_count += 1
        self.instance = 0
        self.name = (name
                     if name
                     else 'convbnrelu{0}'.format(ConvBnRelu.global_count))
        self.conv = lbann.modules.Convolution2dModule(out_channels,
                                                      kernel_size,
                                                      stride=stride,
                                                      padding=padding,
                                                      bias=False,
                                                      name=self.name+'_conv')
        self.bn = BatchNormModule(statistics_group_size=statistics_group_size,
                                  name=self.name+'_bn')

    def forward(self, x):
        self.instance += 1
        x = self.conv(x)
        x = self.bn(x)
        return lbann.Relu(x, name='{0}_relu_instance{1}'.format(self.name, self.instance))

class FcBnRelu(lbann.modules.Module):

    global_count = 0    # Static counter, used for default names

    def __init__(self,
                 size,
                 statistics_group_size=1,
                 name=None,
                 data_layout='data_parallel'):
        super().__init__()
        FcBnRelu.global_count += 1
        self.instance = 0
        self.name = (name
                     if name
                     else 'fcbnrelu{0}'.format(FcBnRelu.global_count))
        self.data_layout = data_layout
        self.fc = lbann.modules.FullyConnectedModule(size,
                                                     bias=False,
                                                     name=self.name+'_fc',
                                                     data_layout=self.data_layout)

        # Weights for batchnorm
        scalebias_vals = [1.0] * size + [0.0] * size
        self.bn_weights = [
            lbann.Weights(
                name='{0}_bn_running_mean'.format(self.name),
                initializer=lbann.ConstantInitializer(value=0.0)),
            lbann.Weights(
                name='{0}_bn_running_var'.format(self.name),
                initializer=lbann.ConstantInitializer(value=1.0)),
            lbann.Weights(
                name='{0}_bn_scalebias'.format(self.name),
                initializer=lbann.ValueInitializer(values=' '.join([str(x) for x in scalebias_vals])))]

    def forward(self, x):
        self.instance += 1
        x = self.fc(x)
        x = lbann.EntrywiseBatchNormalization(x,
                                              weights=[self.bn_weights[0], self.bn_weights[1]],
                                              decay=0.9,
                                              epsilon=1e-5,
                                              name='{0}_bn_instance{1}'.format(self.name, self.instance),
                                              data_layout=self.data_layout)
        x = lbann.EntrywiseScaleBias(x,
                                     weights=self.bn_weights[2],
                                     name='{0}_bn_scalebias_instance{1}'.format(self.name, self.instance),
                                     data_layout=self.data_layout)
        return lbann.Relu(x,
                          name='{0}_relu_instance{1}'.format(self.name, self.instance),
                          data_layout=self.data_layout)

class AlexNetCNN(lbann.modules.Module):
    """AlexNet CNN with batch norm.

    FC network at end of AlexNet is not included.

    """

    def __init__(self, bn_statistics_group_size=1):
        self.name = 'alexnet'
        self.conv1 = ConvBnRelu(96, 11,
                                stride=4,
                                padding=5,
                                statistics_group_size=bn_statistics_group_size,
                                name='{0}_conv1'.format(self.name))
        self.conv2 = ConvBnRelu(256, 3,
                                padding=1,
                                statistics_group_size=bn_statistics_group_size,
                                name='{0}_conv2'.format(self.name))
        self.conv3 = ConvBnRelu(384, 3,
                                padding=1,
                                statistics_group_size=bn_statistics_group_size,
                                name='{0}_conv3'.format(self.name))
        self.conv4 = ConvBnRelu(384, 3,
                                padding=1,
                                statistics_group_size=bn_statistics_group_size,
                                name='{0}_conv4'.format(self.name))
        self.conv5 = ConvBnRelu(256, 3,
                                padding=1,
                                statistics_group_size=bn_statistics_group_size,
                                name='{0}_conv5'.format(self.name))

    def forward(self, x):
        x = self.conv1(x)
        x = lbann.Pooling(x, num_dims=2, has_vectors=False,
                          pool_dims_i=3, pool_pads_i=0, pool_strides_i=2,
                          pool_mode='max')
        x = self.conv2(x)
        x = lbann.Pooling(x, num_dims=2, has_vectors=False,
                          pool_dims_i=3, pool_pads_i=0, pool_strides_i=2,
                          pool_mode='max')
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = lbann.Pooling(x, num_dims=2, has_vectors=False,
                          pool_dims_i=3, pool_pads_i=0, pool_strides_i=2,
                          pool_mode='max')
        return x

class ResNet(lbann.modules.Module):

    def __init__(self, bn_statistics_group_size=1):
        self.name = 'resnet'
        self.cnn = resnet.ResNet34(bn_statistics_group_size=bn_statistics_group_size,
                                   name=self.name)

    def forward(self, x):
        x = self.cnn(x)
        x = lbann.ChannelwiseMean(x)
        return x
