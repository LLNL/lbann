"""Basic neural network building blocks.

These are a convenience for common paradigms built on top of basic layers.

"""

from . import lbann_proto as lp

class SeparableConvolution2d(lp.Module):
    """Depthwise-separable convolution."""

    def __init__(self, name, data_layout, in_channels, out_channels,
                 kernel_size, stride=1, padding=0, dilation=1, bias=True):
        lp.Module.__init__(self, name, data_layout)
        self.conv = lp.Convolution(name + '_conv', data_layout, num_dims=2,
                                   num_output_channels=in_channels,
                                   num_groups=in_channels,
                                   has_vectors=False,
                                   conv_dims_i=kernel_size,
                                   conv_pads_i=padding,
                                   conv_strides_i=stride,
                                   conv_dilations_i=dilation,
                                   has_bias=bias)
        self.pointwise = lp.Convolution(name + '_point', data_layout, num_dims=2,
                                        num_output_channels=out_channels,
                                        has_vectors=False,
                                        conv_dims_i=1,
                                        conv_pads_i=0,
                                        conv_strides_i=1,
                                        conv_dilations_i=1,
                                        has_bias=bias)

    def __call__(self, parent):
        return self.pointwise(self.conv(parent))

class ConvBNRelu2d(lp.Module):
    """Convolution -> Batch normalization -> ReLU"""

    def __init__(self, name, data_layout, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, bias=False, relu=True, bn_global=False):
        lp.Module.__init__(self, name, data_layout)
        self.conv = lp.Convolution(name + '_conv', data_layout, num_dims=2,
                                   num_output_channels=out_channels,
                                   has_vectors=False,
                                   conv_dims_i=kernel_size,
                                   conv_pads_i=padding,
                                   conv_strides_i=stride,
                                   conv_dilations_i=dilation,
                                   has_bias=bias)
        self.bn = lp.BatchNormalization(name + '_bn', data_layout,
                                        decay=0.9, epsilon=1e-5,
                                        global_stats=bn_global)
        if relu:
            self.relu = lp.Relu(name + '_relu', data_layout)
        else:
            self.relu = None

    def __call__(self, parent):
        x = self.bn(self.conv(parent))
        if self.relu is not None:
            x = self.relu(x)
        return x

class ResBottleneck(lp.Module):
    """ResNet bottleneck building block."""

    def __init__(self, name, data_layout, mid_channels, out_channels, stride,
                 dilation=1, downsample=False, bn_global=False):
        lp.Module.__init__(self, name, data_layout)
        self.conv1 = ConvBNRelu2d(name + '_conv1', data_layout, mid_channels, 1,
                                  stride=1, padding=0, dilation=1,
                                  bn_global=bn_global)
        self.conv2 = ConvBNRelu2d(name + '_conv2', data_layout, mid_channels, 3,
                                  stride=stride, padding=dilation, dilation=dilation,
                                  bn_global=bn_global)
        self.conv3 = ConvBNRelu2d(name + '_conv3', data_layout, out_channels, 1,
                                  stride=1, padding=0, dilation=1, relu=False,
                                  bn_global=bn_global)
        if downsample:
            self.downsample = ConvBNRelu2d(name + '_proj', data_layout,
                                           out_channels, 1, stride=stride,
                                           padding=0, dilation=1, relu=False,
                                           bn_global=bn_global)
        else:
            self.downsample = None
        self.sum = lp.Sum(name + '_sum', data_layout)
        self.relu = lp.Relu(name + '_relu', data_layout)
        
    def __call__(self, parent):
        residual = parent
        x = self.conv3(self.conv2(self.conv1(parent)))
        if self.downsample is not None:
            residual = self.downsample(residual)
        return self.relu(self.sum(x)(residual))

class ResBlock(lp.Module):
    """ResNet block, constructed of some number of bottleneck layers."""

    def __init__(self, name, data_layout, num_layers, mid_channels,
                 out_channels, stride, dilation=1, bn_global=False):
        lp.Module.__init__(self, name, data_layout)
        self.layers = []
        self.layers.append(ResBottleneck(
            name + '_bottleneck1', data_layout, mid_channels, out_channels,
            stride, dilation=dilation, downsample=True, bn_global=bn_global))
        for i in range(num_layers - 1):
            self.layers.append(ResBottleneck(
                name + '_bottleneck{0}'.format(i+2), data_layout, mid_channels,
                out_channels, stride=1, dilation=dilation, downsample=False,
                bn_global=bn_global))

    def __call__(self, parent):
        x = parent
        for l in self.layers:
            x = l(x)
        return x
