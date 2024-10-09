import lbann
import lbann.modules

class PROBIESNetLBANN(lbann.modules.Module):

    global_count = 0  # Static counter, used for default names

    def __init__(self, output_size, intermed_fc_layers, activation, dropout_percent, name=None):
        """Initialize PROBIESNet.

        Args:
            output_size (int): Size of output tensor.
            name (str, optional): Module name
                (default: 'probiesnet_module<index>').

        """
        PROBIESNetLBANN.global_count += 1
        self.instance = 0
        self.intermed_fc_layers = intermed_fc_layers 
        self.dropout_percent = dropout_percent
        self.name = (name if name
                     else 'probiesNet_module{0}'.format(PROBIESNetLBANN.global_count))
        conv = lbann.modules.Convolution2dModule
        fc = lbann.modules.FullyConnectedModule
        self.conv1 = conv(36, 11, stride=4, activation=activation, 
                          name=self.name+'_conv1')
        self.conv2 = conv(64, 5, padding=2, activation=activation,
                          name=self.name+'_conv2')
        for idx,layer in enumerate(intermed_fc_layers):
            setattr(self, 'fc' + str(idx), fc(layer, activation=activation, name=self.name+'_fc'+str(idx)))
        setattr(self, 'fc' + str((len(self.intermed_fc_layers))+1), fc(output_size, name='pred_out'))

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
        for idx, layer in enumerate(self.intermed_fc_layers):
            x = getattr(self, 'fc' + str(idx))(x)
            x = lbann.Dropout(x, keep_prob=self.dropout_percent,
                          name='{0}_drop_search_' + str(idx) + '_instance{1}'.format(self.name,self.instance))
            x = lbann.Relu(x)
        return getattr(self, 'fc' + str(len(self.intermed_fc_layers)+1))(x)
