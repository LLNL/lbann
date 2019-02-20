"""Neural network modules.

These are a convenience for common layer patterns that are often the
basic building blocks for larger models.

"""

import lbann.proto as lp
from collections.abc import Iterable
import warnings

def _make_iterable(obj):
    """Convert to an iterable object.

    Simply returns `obj` if it is alredy iterable. Otherwise returns a
    1-tuple containing `obj`.

    """
    if isinstance(obj, Iterable):
        return obj
    else:
        return (obj,)

class Module:
    """Base class for neural network modules.

    A module is a pattern of operations that may be applied to a set
    of input layers, obtaining a set of output layers.

    """

    def __init__(self):
        pass

    def forward(self, input):
        """Apply module pattern to `input`.

        `input` is a `Layer` or a sequence of `Layer`s. The module
        pattern is added to the layer graph and the output layer(s)
        are returned.

        """
        # Should be overridden in all sub-classes
        raise NotImplementedError

    def __call__(self, input):
        """Apply module mattern to `input`.

        Syntatic sugar around `forward` function.

        """
        return self.forward(input)

class FullyConnectedModule(Module):
    """Basic block for fully-connected neural networks.

    Applies a dense linearity and a nonlinear activation function.

    """

    global_count = 0  # Static counter, used for default names

    def __init__(self, size, bias=True, weights=[], activation=None,
                 name=None, data_layout='data_parallel'):
        """Initialize fully-connected module.

        Args:
            size (int): Size of output tensor.
            activation (type): Layer class for activation function.
            bias (bool): Whether to apply bias after linearity.
            weights (`Weights` or iterator of `Weights`): Weights in
                fully-connected layer. There are at most two: the matrix
                and the bias. If weights are not provided, LBANN will
                initialize them with default settings.
            name (str): Default name is in the form 'fcmodule<index>'.
            data_layout (str): Data layout.

        """
        super().__init__()
        FullyConnectedModule.global_count += 1
        self.instance = 0
        self.size = size
        self.bias = bias
        self.name = (name
                     if name
                     else 'fcmodule{0}'.format(FullyConnectedModule.global_count))
        self.data_layout = data_layout

        # Initialize weights
        # Note: If weights are not provided, matrix weights are
        # initialized with He normal scheme and bias weights are
        # initialized with zeros.
        self.weights = list(_make_iterable(weights))
        if len(self.weights) > 2:
            raise ValueError('`FullyConnectedModule` has '
                             'at most two weights, '
                             'but got {0}'.format(len(self.weights)))
        if len(self.weights) == 0:
            self.weights.append(
                lp.Weights(initializer=lp.HeNormalInitializer(),
                           name=self.name+'_matrix'))
        if len(self.weights) == 1:
            self.weights.append(
                lp.Weights(initializer=lp.ConstantInitializer(value=0.0),
                           name=self.name+'_bias'))

        # Initialize activation layer
        self.activation = None
        if activation:
            if isinstance(activation, type):
                self.activation = activation
            else:
                self.activation = type(activation)
            if not issubclass(self.activation, lp.Layer):
                raise ValueError('activation must be a layer')

    def forward(self, x):
        self.instance += 1
        name = '{0}_instance{1}'.format(self.name, self.instance)
        y = lp.FullyConnected(x,
                              weights=self.weights,
                              name=(name+'_fc' if self.activation else name),
                              data_layout=self.data_layout,
                              num_neurons=self.size,
                              has_bias=self.bias)
        if self.activation:
            return self.activation(y,
                                   name=name+'_activation',
                                   data_layout=self.data_layout)
        else:
            return y

class ConvolutionNdModule(Module):
    """Basic block for ND convolutional neural networks.

    Applies a convolution and a nonlinear activation function.

    """

    global_count = 0  # Static counter, used for default names

    def __init__(self, num_dims,
                 out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 weights=[], activation=None, name=None):
        """Initialize convolution module.

        Args:
            num_dims (int): Number of dimensions.
            out_channels (int): Number of output channels, i.e. number
                of filters.
            kernel_size (int): Size of convolution kernel.
            stride (int): Convolution stride.
            padding (int): Convolution padding.
            dilation (int): Convolution dilation.
            groups (int): Number of convolution groups.
            bias (bool): Whether to apply channel-wise bias after
                convolution.
            weights (`Weights` or iterator of `Weights`): Weights in
                convolution layer. There are at most two: the kernel
                and the bias. If weights are not provided, LBANN will
                initialize them with default settings.
            name (str): Default name is in the form 'convmodule<index>'.

        """
        super().__init__()
        ConvolutionNdModule.global_count += 1
        self.instance = 0
        self.num_dims = num_dims
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.weights = list(_make_iterable(weights))
        self.name = (name
                     if name
                     else 'convmodule{0}'.format(ConvolutionNdModule.global_count))

        # Initialize weights
        # Note: If weights are not provided, kernel weights are
        # initialized with He normal scheme and bias weights are
        # initialized with zeros.
        self.weights = list(_make_iterable(weights))
        if len(self.weights) > 2:
            raise ValueError('`ConvolutionNdModule` has '
                             'at most two weights, '
                             'but got {0}'.format(len(self.weights)))
        if len(self.weights) == 0:
            self.weights.append(
                lp.Weights(initializer=lp.HeNormalInitializer(),
                           name=self.name+'_kernel'))
        if len(self.weights) == 1:
            self.weights.append(
                lp.Weights(initializer=lp.ConstantInitializer(value=0.0),
                           name=self.name+'_bias'))

        # Initialize activation layer
        self.activation = None
        if activation:
            if isinstance(activation, type):
                self.activation = activation
            else:
                self.activation = type(activation)
            if not issubclass(self.activation, lp.Layer):
                raise ValueError('activation must be a layer')

    def forward(self, x):
        self.instance += 1
        name = '{0}_instance{1}'.format(self.name, self.instance)
        y = lp.Convolution(x,
                           weights=self.weights,
                           name=(name+'_conv' if self.activation else name),
                           num_dims=self.num_dims,
                           num_output_channels=self.out_channels,
                           has_vectors=False,
                           conv_dims_i=self.kernel_size,
                           conv_pads_i=self.padding,
                           conv_strides_i=self.stride,
                           conv_dilations_i=self.dilation,
                           num_groups=self.groups,
                           has_bias=self.bias)
        if self.activation:
            return self.activation(y, name=name+'_activation')
        else:
            return y

class Convolution2dModule(ConvolutionNdModule):
    """Basic block for 2D convolutional neural networks.

    Applies a convolution and a nonlinear activation function.
    This is a wrapper class for ConvolutionNdModule.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(2, *args, **kwargs)

class Convolution3dModule(ConvolutionNdModule):
    """Basic block for 3D convolutional neural networks.

    Applies a convolution and a nonlinear activation function.
    This is a wrapper class for ConvolutionNdModule.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(3, *args, **kwargs)
