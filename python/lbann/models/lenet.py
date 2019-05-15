import lbann
import lbann.modules

class LeNet(lbann.modules.Module):
    """LeNet neural network.

    Assumes image data in NCHW format.

    See:
        Yann LeCun, Leon Bottou, Yoshua Bengio, and Patrick
        Haffner. "Gradient-based learning applied to document
        recognition." Proceedings of the IEEE 86, no. 11 (1998):
        2278-2324.

    """

    global_count = 0  # Static counter, used for default names

    def __init__(self, output_size, name=None):
        """Initialize LeNet.

        Args:
            output_size (int): Size of output tensor.
            name (str, optional): Module name
                (default: 'lenet_module<index>').

        """
        LeNet.global_count += 1
        self.instance = 0
        self.name = (name if name
                     else 'lenet_module{0}'.format(LeNet.global_count))
        conv = lbann.modules.Convolution2dModule
        fc = lbann.modules.FullyConnectedModule
        self.conv1 = conv(6, 5, activation=lbann.Relu,
                          name=self.name+'_conv1')
        self.conv2 = conv(16, 5, activation=lbann.Relu,
                          name=self.name+'_conv2')
        self.fc1 = fc(120, activation=lbann.Relu, name=self.name+'_fc1')
        self.fc2 = fc(84, activation=lbann.Relu, name=self.name+'_fc2')
        self.fc3 = fc(output_size, name=self.name+'_fc3')

    def forward(self, x):
        self.instance += 1

        # Convolutional network
        x = self.conv1(x)
        x = lbann.Pooling(x, num_dims=2, has_vectors=False,
                          pool_dims_i=2, pool_pads_i=0, pool_strides_i=2,
                          pool_mode='max',
                          name='{0}_pool1_instance{1}'.format(self.name,self.instance))
        x = self.conv2(x)
        x = lbann.Pooling(x, num_dims=2, has_vectors=False,
                       pool_dims_i=2, pool_pads_i=0, pool_strides_i=2,
                       pool_mode='max',
                       name='{0}_pool2_instance{1}'.format(self.name,self.instance))
        return self.fc3(self.fc2(self.fc1(x)))
