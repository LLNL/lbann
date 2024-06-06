import lbann
import lbann.models
import lbann.modules as lm
import numpy as np
import math
from functools import partial

class CosmoFlow(lm.Module):
    """The CosmoFlow neural network.

    See:
        Amrita Mathuriya, Deborah Bard, Peter Mendygral, Lawrence Meadows,
        James Arnemann, Lei Shao, Siyu He, Tuomas Karna, Diana Moise,
        Simon J. Pennycook, Kristyn Maschhoff, Jason Sewall, Nalini Kumar,
        Shirley Ho, Michael F. Ringenburg, Prabhat, and Victor Lee.
        "Cosmoflow: Using deep learning to learn the universe at scale."
        Proceedings of the International Conference for High Performance
        Computing, Networking, Storage, and Analysis, SC'18, pp. 65:1-65:11,
        2018.
    """

    global_count = 0  # Static counter, used for default names

    def __init__(self,
                 input_width,
                 output_size,
                 name=None,
                 use_bn=False,
                 bn_statistics_group_size=None,
                 mlperf=False,
                 transform_input=False,
                 dropout_keep_prob=0.5):
        """Initialize CosmFlow.

        Args:
            input_width (int): Size of each spatial dimension of input data.
            output_size (int): Size of output tensor.
            name (str, optional): Module name
                (default: 'cosmoflow_module<index>').
            use_bn (bool): Whether or not batch normalization layers are used.
            bn_statistics_group_size (int): The number of samples
                for each batch-normalization group.
            mlperf (bool): Whether or not to use the MLPerf HPC compliant 
                model.
            transform_input (bool): Whether or not to apply log1p
                transformation to model inputs.
            dropout_keep_prob (float): Probability of not zeroing out
                activations in dropout layers. Setting to 1 disables dropout.
        """

        CosmoFlow.global_count += 1
        self.instance = 0
        self.name = (name if name
                     else 'cosmoflow_module{0}'.format(CosmoFlow.global_count))
        self.input_width = input_width
        self.use_bn = use_bn
        self.mlperf = mlperf
        self.transform_input = transform_input
        self.dropout_keep_prob = dropout_keep_prob

        if self.mlperf:
            base_channels = 32
            max_channels = 512
        else:
            base_channels = 16
            max_channels = 256

        assert self.input_width in [128, 256, 512]
        num_conv_layers = int(math.log2(self.input_width)) - 2
        self.cp_params = []
        for i in range(num_conv_layers):
            out_channels = min(base_channels * 2**i, max_channels)
            self.cp_params += [
                {"type": "conv", "out_channels": out_channels},
                {"type": "pool"}
            ]

        for p in self.cp_params:
            if p["type"] == "conv":
                p["kernel_size"] = 3
                p["padding"] = 1
                p["stride"] = 1

        # Create convolutional blocks
        activation = partial(lbann.LeakyRelu, negative_slope=0.3)
        for i, param in enumerate(filter(lambda x: x["type"] == "conv", self.cp_params)):
            conv_name ="conv"+str(i+1)
            conv_weights = [lbann.Weights(
                initializer=lbann.HeNormalInitializer())]
            if not use_bn:
                conv_weights += [
                    lbann.Weights(initializer=lbann.ConstantInitializer(value=0))
                ]
            param_actual = dict(param)
            param_actual.pop("type", None)
            conv = ConvBNRelu(
                **param_actual,
                conv_weights=conv_weights,
                use_bn=self.use_bn,
                bn_statistics_group_size=bn_statistics_group_size,
                bn_zero_init=False,
                name=self.name+"_"+conv_name,
                activation=activation)
            setattr(self, conv_name, conv)

        # Create fully-connected layers
        fc_params = [
            {"size": 128 if mlperf else 2048},
            {"size": 64 if mlperf else 256},
            {"size": output_size},
        ]
        for i, param in enumerate(fc_params):
            fc_name = "fc"+str(i+1)
            fc = lm.FullyConnectedModule(
                **param,
                name=self.name+"_"+fc_name,
                weights=[lbann.Weights(initializer=lbann.HeNormalInitializer() if i < len(fc_params)-1 else lbann.ConstantInitializer(value=0)),
                         lbann.Weights(initializer=lbann.ConstantInitializer(value=0))],
            )
            setattr(self, fc_name, fc)

    def forward(self, x):
        self.instance += 1

        if self.transform_input:
            x = lbann.Log1p(x, name=f'{self.name}_input_transform_instance{self.instance}')

        def create_pooling(x, i):
            if self.mlperf:
                return lbann.Pooling(
                    x, num_dims=3, has_vectors=False,
                    pool_dims_i=2, pool_pads_i=0, pool_strides_i=2,
                    pool_mode='max',
                    name='{0}_pool{1}_instance{2}'.format(
                        self.name, i, self.instance))
            else:
                return lbann.Pooling(
                    x, num_dims=3, has_vectors=False,
                    pool_dims_i=3, pool_pads_i=1, pool_strides_i=2,
                    pool_mode='average',
                    name='{0}_pool{1}_instance{2}'.format(
                        self.name, i, self.instance))
            
        def create_act(x, i):
            return lbann.LeakyRelu(
                x, negative_slope=0.3,
                name='{0}_fc_act{1}_instance{2}'.format(
                    self.name, i, self.instance))

        def create_dropout(x, i):
            if self.dropout_keep_prob == 1:
                return x
            
            return lbann.Dropout(
                x, keep_prob=self.dropout_keep_prob,
                name='{0}_fc_drop{1}_instance{2}'.format(
                    self.name, i, self.instance))

        # Convolutional blocks
        i_conv = 1
        i_pool = 1
        for param in self.cp_params:
            if param["type"] == "conv":
                x = getattr(self, "conv{}".format(i_conv))(x)
                i_conv += 1

            else:
                x = create_pooling(x, i_pool)
                i_pool += 1

        # Fully-connected layers
        for i in range(3):
            if i > 0:
                x = create_act(x, i)
                x = create_dropout(x, i)
            x = getattr(self, "fc{}".format(i+1))(x)

        x = lbann.Scale(lbann.Tanh(x), constant=1.2)

        return x


class ConvBNRelu(lbann.modules.Module):
    """Convolution -> Batch normalization -> ReLU

    Adapted from ResNets. Assumes image data in NCDHW format.
    """

    def __init__(self, out_channels, kernel_size, stride, padding,
                 use_bn, bn_zero_init, bn_statistics_group_size,
                 activation, name,
                 conv_weights):
        """Initialize ConvBNRelu module.

        Args:
            out_channels (int): Number of output channels, i.e. number
                of convolution filters.
            kernel_size (int): Size of convolution kernel.
            stride (int): Convolution stride.
            padding (int): Convolution padding.
            use_bn (bool): Whether or not batch normalization layers are used.
            bn_zero_init (bool): Zero-initialize batch normalization
                scale.
            bn_statistics_group_size (int): Aggregation size for batch
                normalization statistics.
            activation (lbann.Layer): The activation function.
            name (str): Module name.
            conv_weights (lbann.Weights): Pre-defined weights.
        """

        super().__init__()
        self.name = name
        self.instance = 0
        self.bn_statistics_group_size = bn_statistics_group_size
        self.activation = activation
        self.use_bn = use_bn
        self.conv_weights = conv_weights

        # Initialize convolution
        self.conv = lbann.modules.Convolution3dModule(
            out_channels, kernel_size,
            stride=stride, padding=padding,
            bias=(not use_bn), weights=self.conv_weights,
            name=self.name + '_conv')

        # Initialize batch normalization
        if self.use_bn:
            bn_scale_init = 0.0 if bn_zero_init else 1.0
            bn_scale = lbann.Weights(
                initializer=lbann.ConstantInitializer(value=bn_scale_init),
                name=self.name + '_bn_scale')
            bn_bias = lbann.Weights(
                initializer=lbann.ConstantInitializer(value=0.0),
                name=self.name + '_bn_bias')
            self.bn_weights = [bn_scale, bn_bias]

    def forward(self, x):
        self.instance += 1
        layer = self.conv(x)
        if self.use_bn:
            layer = lbann.BatchNormalization(
                layer, weights=self.bn_weights,
                statistics_group_size=self.bn_statistics_group_size,
                decay=0.99,
                name='{0}_bn_instance{1}'.format(
                    self.name, self.instance))
        if self.activation:
            layer = self.activation(
                layer,
                negative_slope=0.3,
                name='{0}_activation_instance{1}'.format(
                    self.name, self.instance))
        return layer
