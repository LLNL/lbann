import lbann
import lbann.modules

class AlexNet(lbann.modules.Module):
    """AlexNet neural network.

    Assumes image data in NCHW format.

    See:
        Alex Krizhevsky, Ilya Sutskever, and Geoffrey
        E. Hinton. "ImageNet classification with deep convolutional
        neural networks." In Advances in Neural Information Processing
        Systems, pp. 1097-1105. 2012.

    Note that there is very little consistency in the implementation of
    AlexNet across frameworks. If a particular variant is needed, you should
    implement it yourself.

    """

    global_count = 0  # Static counter, used for default names

    def __init__(self, output_size, name=None):
        """Initialize AlexNet.

        Args:
            output_size (int): Size of output tensor.
            name (str, optional): Module name
                (default: 'alexnet_module<index>').

        """
        AlexNet.global_count += 1
        self.instance = 0
        self.name = (name if name
                     else 'alexnet_module{0}'.format(AlexNet.global_count))
        conv = lbann.modules.Convolution2dModule
        fc = lbann.modules.FullyConnectedModule
        self.conv1 = conv(96, 11, stride=4, activation=lbann.Relu,
                          name=self.name+'_conv1')
        self.conv2 = conv(256, 5, padding=2, activation=lbann.Relu,
                          name=self.name+'_conv2')
        self.conv3 = conv(384, 3, padding=1, activation=lbann.Relu,
                          name=self.name+'_conv3')
        self.conv4 = conv(384, 3, padding=1, activation=lbann.Relu,
                          name=self.name+'_conv4')
        self.conv5 = conv(256, 3, padding=1, activation=lbann.Relu,
                          name=self.name+'_conv5')
        self.fc6 = fc(4096, activation=lbann.Relu, name=self.name+'_fc6')
        self.fc7 = fc(4096, activation=lbann.Relu, name=self.name+'_fc7')
        self.fc8 = fc(output_size, name=self.name+'_fc8')

    def forward(self, x):
        self.instance += 1

        # Convolutional network
        x = self.conv1(x)
        x = lbann.LocalResponseNormalization(
            x, window_width=5, lrn_alpha=0.0001, lrn_beta=0.75, lrn_k=2,
            name='{0}_norm1_instance{1}'.format(self.name,self.instance))
        x = lbann.Pooling(x, num_dims=2, has_vectors=False,
                          pool_dims_i=3, pool_pads_i=0, pool_strides_i=2,
                          pool_mode='max',
                          name='{0}_pool1_instance{1}'.format(self.name,self.instance))
        x = self.conv2(x)
        x = lbann.LocalResponseNormalization(
            x, window_width=5, lrn_alpha=0.0001, lrn_beta=0.75, lrn_k=2,
            name='{0}_norm2_instance{1}'.format(self.name,self.instance))
        x = lbann.Pooling(x, num_dims=2, has_vectors=False,
                          pool_dims_i=3, pool_pads_i=0, pool_strides_i=2,
                          pool_mode='max',
                          name='{0}_pool2_instance{1}'.format(self.name,self.instance))
        x = self.conv5(self.conv4(self.conv3(x)))
        x = lbann.Pooling(x, num_dims=2, has_vectors=False,
                          pool_dims_i=3, pool_pads_i=0, pool_strides_i=2,
                          pool_mode='max',
                          name='{0}_pool5_instance{1}'.format(self.name,self.instance))

        # Fully-connected network
        x = self.fc6(x)
        x = lbann.Dropout(x, keep_prob=0.5,
                          name='{0}_drop6_instance{1}'.format(self.name,self.instance))
        x = self.fc7(x)
        x = lbann.Dropout(x, keep_prob=0.5,
                          name='{0}_drop7_instance{1}'.format(self.name,self.instance))
        return self.fc8(x)
