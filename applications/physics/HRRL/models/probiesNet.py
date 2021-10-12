import lbann
import lbann.modules

class PROBIESNet(lbann.modules.Module):

    global_count = 0  # Static counter, used for default names

    def __init__(self, output_size, name=None):
        """Initialize PROBIESNet.

        Args:
            output_size (int): Size of output tensor.
            name (str, optional): Module name
                (default: 'probiesnet_module<index>').

        """
        PROBIESNet.global_count += 1
        self.instance = 0
        self.name = (name if name
                     else 'probiesNet_module{0}'.format(PROBIESNet.global_count))
        conv = lbann.modules.Convolution2dModule
        fc = lbann.modules.FullyConnectedModule
        self.conv1 = conv(36, 11, stride=4, activation=lbann.Relu,
                          name=self.name+'_conv1')
        self.conv2 = conv(64, 5, padding=2, activation=lbann.Relu,
                          name=self.name+'_conv2')
        self.fc1 = fc(480, activation=lbann.Relu, name=self.name+'_fc1')
        self.fc2 = fc(240, activation=lbann.Relu, name=self.name+'_fc2')
        self.fc3 = fc(output_size, name='pred')

    def forward(self, x):
        self.instance += 1

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

        x = self.fc1(x)
        x = lbann.Dropout(x, keep_prob=0.5,
                          name='{0}_drop6_instance{1}'.format(self.name,self.instance))
        x = self.fc2(x)
        x = lbann.Dropout(x, keep_prob=0.5,
                          name='{0}_drop7_instance{1}'.format(self.name,self.instance))
        return self.fc3(x)
