import lbann
import lbann.models
import lbann.modules as lm
import numpy as np

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
                 bn_statistics_group_size=None):
        """Initialize CosmFlow.

        Args:
            input_width (int): Size of each spatial dimension of input data.
            output_size (int): Size of output tensor.
            name (str, optional): Module name
                (default: 'cosmoflow_module<index>').
            use_bn (bool): Whether or not batch normalization layers are used.
            bn_statistics_group_size (int): The number of samples
                for each batch-normalization group.
        """

        CosmoFlow.global_count += 1
        self.instance = 0
        self.name = (name if name
                     else 'cosmoflow_module{0}'.format(CosmoFlow.global_count))
        self.input_width = input_width
        self.use_bn = use_bn

        assert self.input_width in [128, 256, 512]
        self.cp_params = [
            {"type": "conv", "out_channels": 16,  "kernel_size": 3, "stride": 1},
            {"type": "pool"},
            {"type": "conv", "out_channels": 32,  "kernel_size": 3, "stride": 1},
            {"type": "pool"},
            {"type": "conv", "out_channels": 64,  "kernel_size": 3, "stride": 1},
            {"type": "pool"},
            {"type": "conv", "out_channels": 128, "kernel_size": 3, "stride": 2},
            {"type": "pool"},
            {"type": "conv", "out_channels": 256, "kernel_size": 3, "stride": 1},
            {"type": "pool"},
            {"type": "conv", "out_channels": 256, "kernel_size": 3, "stride": 1},
            {"type": "conv", "out_channels": 256, "kernel_size": 3, "stride": 1},
        ]
        additional_pools = []
        if self.input_width == 256:
            additional_pools = [6]
        elif self.input_width == 512:
            additional_pools = [6, 7]

        for i in additional_pools:
            conv_idx = list(np.cumsum([1 if x["type"] == "conv" else
                                       0 for x in self.cp_params])).index(i)
            self.cp_params.insert(conv_idx+1, {"type": "pool"})

        for p in self.cp_params:
            if p["type"] == "conv":
                p["padding"] = int((p["kernel_size"]-1)/2)

        # Create convolutional blocks
        activation = lbann.LeakyRelu
        for i, param in enumerate(filter(lambda x: x["type"] == "conv", self.cp_params)):
            conv_name ="conv"+str(i+1)
            conv_weights = [lbann.Weights(
                initializer=lbann.GlorotUniformInitializer())]
            param_actual = dict(param)
            param_actual.pop("type", None)
            conv = ConvBNRelu(
                **param_actual,
                conv_weights=conv_weights,
                use_bn=self.use_bn,
                bn_statistics_group_size=bn_statistics_group_size,
                bn_zero_init=False,
                name=self.name+"_"+conv_name,
                activation=lbann.LeakyRelu)
            setattr(self, conv_name, conv)

        # Create fully-connected layers
        fc_params = [
            {"size": 2048},
            {"size": 256},
            {"size": output_size},
        ]
        for i, param in enumerate(fc_params):
            fc_name = "fc"+str(i+1)
            fc = lm.FullyConnectedModule(
                **param,
                activation=activation if i < len(fc_params)-1 else None,
                name=self.name+"_"+fc_name,
                weights=[lbann.Weights(initializer=lbann.GlorotUniformInitializer()),
                         lbann.Weights(initializer=lbann.ConstantInitializer(value=0.1))],
            )
            setattr(self, fc_name, fc)

    def forward(self, x):
        self.instance += 1

        def create_pooling(x, i):
            return lbann.Pooling(
                x, num_dims=3, has_vectors=False,
                pool_dims_i=3, pool_pads_i=1, pool_strides_i=2,
                pool_mode='average',
                name='{0}_pool{1}_instance{2}'.format(
                    self.name, i, self.instance))

        def create_dropout(x, i):
            return lbann.Dropout(
                x, keep_prob=0.8,
                name='{0}_drop{1}_instance{2}'.format(
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
            x = create_dropout(x, i+1)
            x = getattr(self, "fc{}".format(i+1))(x)

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
            bias=False, weights=self.conv_weights,
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
                decay=0.999,
                name='{0}_bn_instance{1}'.format(
                    self.name, self.instance))
        if self.activation:
            layer = self.activation(
                layer,
                name='{0}_activation_instance{1}'.format(
                    self.name, self.instance))
        return layer
