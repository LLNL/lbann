import lbann
import lbann.models
import lbann.modules as lm


class UNet3D(lm.Module):
    """The 3D U-Net.

    See:
    \"{O}zg\"{u}n \c{C}i\c{c}ek, Ahmed Abdulkadir, Soeren S. Lienkamp,
    Thomas Brox, and Olaf Ronneberger. "3D U-Net: learning dense volumetric
    segmentation from sparse annotation." In International conference
    on medical image computing and computer-assisted intervention,
    pp. 424-432, 2016.

    Note that this model assumes the same spatial input/output sizes with
    extra padding to simplify the implementation.
    """

    global_count = 0  # Static counter, used for default names

    def __init__(self, name=None):
        """Initialize 3D U-Net.

        Args:
            name (str, optional): Module name
                (default: 'alexnet_module<index>').
        """

        UNet3D.global_count += 1
        self.instance = 0
        self.name = (name if name
                     else "unet3d_module{0}".format(UNet3D.global_count))

        # The list of ([down-conv filters], [up-conv filters], deconv filters)
        self.BLOCKS = [
            ([32, 64], [64, 64], 128),  # bconv1_down, bconv3_up, deconv3
            ([64, 128], [128, 128], 256),  # bconv2_down, bconv2_up, deconv2
            ([128, 256], [256, 256], 512),  # bconv3_down, bconv1_up, deconv1
            ]
        # The list of the number of filters of the "bottom" convolution block
        self.BOTTOM_BLOCK = [256, 512]
        # The number of pooling/deconvolution layers
        self.NUM_LEVELS = len(self.BLOCKS)
        # Whether PARTITIONED_LEVELS-th pooling/deconvolution is partitioned
        self.PARTITION_INCLUDE_POOL = True

        # Deconvolution should have the same number of input/output channels
        assert self.BLOCKS[-1][2] == self.BOTTOM_BLOCK[1]
        assert all([self.BLOCKS[x][2] == self.BLOCKS[x+1][1][-1]
                    for x in range(self.NUM_LEVELS-1)])

        # Building blocks
        self.downconvs = []
        self.upconvs = []
        self.deconvs = []
        for i, blocks in enumerate(self.BLOCKS):
            downBlock, upBlock, deconv = blocks
            self.downconvs.append(UNet3DConvBlock(
                downBlock, name="{}_bconv{}_down".format(self.name, i+1)))
            ui = self.NUM_LEVELS-1-i
            self.upconvs.insert(0, UNet3DConvBlock(
                upBlock, name="{}_bconv{}_up".format(self.name, ui+1)))
            self.deconvs.insert(0, Deconvolution3dModule(
                deconv, 2, stride=2, padding=0, activation=None,
                bias=False,
                name="{}_deconv{}".format(self.name, ui+1)))

        # The bottom convolution
        self.bottomconv = UNet3DConvBlock(
            self.BOTTOM_BLOCK, name="{}_bconv_bottom".format(self.name))

        # The last convolution
        self.lastconv = lm.Convolution3dModule(
            3, 1, stride=1, padding=0, activation=None,
            bias=False,
            name="{}_lconv".format(self.name))

    def forward(self, x):
        self.instance += 1

        x_concat = []
        for i in range(self.NUM_LEVELS):
            x = self.downconvs[i](x)
            x_concat.append(x)
            x = lbann.Pooling(
                x, num_dims=3, has_vectors=False,
                pool_dims_i=2, pool_pads_i=0, pool_strides_i=2,
                pool_mode="max",
                name="{}_pool{}_instance{}".format(
                    self.name, i+1, self.instance))

        x = self.bottomconv(x)

        for i in range(self.NUM_LEVELS):
            x = self.deconvs[i](x)
            x = self.upconvs[i](x, x_concat=x_concat[self.NUM_LEVELS-1-i])

        x = self.lastconv(x)
        x = lbann.Softmax(
            x,
            softmax_mode="channel")

        return x


class UNet3DConvBlock(lm.Module):
    """Basic block of an optional concatenation layer and
    a list of 3D convolutional layers.
    """

    def __init__(self, out_channels_list, name):
        super().__init__()
        self.name = name
        self.instance = 0
        assert len(out_channels_list) == 2

        self.convs = []
        for i, channels in enumerate(out_channels_list):
            self.convs.append(Convolution3dBNModule(
                channels,
                3,
                stride=1,
                padding=1,
                activation=lbann.Relu,
                bias=False,
                name="{}_conv_block_{}".format(self.name, i+1)))

    def forward(self, x, x_concat=None):
        self.instance += 1
        if x_concat is not None:
            x = lbann.Concatenation(
                [x, x_concat],
                axis=0)

        for c in self.convs:
            x = c(x)

        return x


class Convolution3dBNModule(lm.Module):
    """Basic block of a batch-normalization layer, 3D convolutional
    layer, and an optional activation layer.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.name = kwargs["name"]
        self.activation = None if "activation" not in kwargs.keys() \
            else kwargs["activation"]
        kwargs["activation"] = None

        self.conv = lm.Convolution3dModule(*args, **kwargs)

        bn_scale = lbann.Weights(
            initializer=lbann.ConstantInitializer(value=1.0),
            name="{}_bn_scale".format(self.name))
        bn_bias = lbann.Weights(
            initializer=lbann.ConstantInitializer(value=0.0),
            name="{}_bn_bias".format(self.name))
        self.bn_weights = [bn_scale, bn_bias]
        self.instance = 0

    def forward(self, x):
        self.instance += 1
        x = self.conv(x)
        x = lbann.BatchNormalization(
            x,
            weights=self.bn_weights,
            statistics_group_size=-1,
            name="{}_bn_instance{}".format(
                self.name,
                self.instance))
        if self.activation is not None:
            x = self.activation(x)

        return x


class Deconvolution3dModule(lm.ConvolutionModule):
    """Basic block for 3D deconvolutional neural networks.

    Applies a deconvolution and a nonlinear activation function.
    This is a wrapper class for ConvolutionModule.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(3, transpose=True, *args, **kwargs)
