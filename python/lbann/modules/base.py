"""Base class for neural network modules.

This also contains modules for fully-connected and convolution layers.

"""
import abc
import lbann
from lbann.util import make_iterable

class Module(abc.ABC):
    """Base class for neural network modules.

    A module is a pattern of layers that can be added to a layer
    graph, possibly multiple times. The pattern typically takes a set
    of input layers and obtains a set of output layers.

    """

    def forward(self, *args, **kwargs):
        """Apply module pattern.

        A module pattern typically takes a set of `Layer`s as input
        and returns a set of `Layer`s.

        """
        # Should be overridden in all sub-classes
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        """Apply module mattern to `input`.

        Syntatic sugar around `forward` function.

        """
        return self.forward(*args, **kwargs)

class FullyConnectedModule(Module):
    """Basic block for fully-connected neural networks.

    Applies a dense linearity and a nonlinear activation function.

    """

    global_count = 0  # Static counter, used for default names

    def __init__(self,
                 size,
                 bias=True,
                 transpose=False,
                 weights=[],
                 activation=None,
                 name=None,
                 channelwise=False,
                 data_layout='data_parallel',
                 parallel_strategy={}):
        """Initialize fully-connected module.

        Args:
            size (int): Size of output tensor.
            activation (type): Layer class for activation function.
            bias (bool): Whether to apply bias after linearity.
            transpose (bool): Whether to apply transpose of weights
                matrix.
            weights (`Weights` or iterator of `Weights`): Weights in
                fully-connected layer. There are at most two: the
                matrix and the bias. If weights are not provided, the
                matrix will be initialized with He normal
                initialization and the bias with zeros.
            name (str): Default name is in the form 'fcmodule<index>'.
            data_layout (str): Data layout.
            parallel_strategy (dict): Data partitioning scheme.

        """
        super().__init__()
        FullyConnectedModule.global_count += 1
        self.instance = 0
        self.size = size
        self.bias = bias
        self.transpose = transpose
        self.channelwise = channelwise
        self.data_layout = data_layout
        self.parallel_strategy = parallel_strategy

        base_name = ('channelwiseFCmodule{0}' 
                     if self.channelwise
                     else 'fcmodule{0}')

        self.name = (name
                     if name
                     else base_name.format(FullyConnectedModule.global_count))

        # Initialize weights
        # Note: If weights are not provided, matrix weights are
        # initialized with He normal scheme and bias weights are
        # initialized with zeros.
        self.weights = list(make_iterable(weights))
        if len(self.weights) > 2:
            raise ValueError('`FullyConnectedModule` has '
                             'at most two weights, '
                             'but got {0}'.format(len(self.weights)))
        if len(self.weights) == 0:
            self.weights.append(
                lbann.Weights(initializer=lbann.HeNormalInitializer(),
                              name=self.name+'_matrix'))
        if len(self.weights) == 1:
            self.weights.append(
                lbann.Weights(initializer=lbann.ConstantInitializer(value=0.0),
                              name=self.name+'_bias'))

        # Initialize activation layer
        self.activation = None
        if activation:
            if isinstance(activation, type):
                self.activation = activation
            else:
                self.activation = type(activation)
            if not issubclass(self.activation, lbann.Layer):
                raise ValueError('activation must be a layer')

    def forward(self, x):
        self.instance += 1
        name = '{0}_instance{1}'.format(self.name, self.instance)
        if (self.channelwise):
          y = lbann.ChannelwiseFullyConnected(x,
                                              weights=self.weights,
                                              name=(name+'_fc' if self.activation else name),
                                              data_layout=self.data_layout,
                                              num_neurons=self.size,
                                              has_bias=self.bias,
                                              transpose=self.transpose,
                                              parallel_strategy=self.parallel_strategy)
        else:
          y = lbann.FullyConnected(x,
                                   weights=self.weights,
                                   name=(name+'_fc' if self.activation else name),
                                   data_layout=self.data_layout,
                                   num_neurons=self.size,
                                   has_bias=self.bias,
                                   transpose=self.transpose,
                                   parallel_strategy=self.parallel_strategy)
        if self.activation:
            return self.activation(y,
                                   name=name+'_activation',
                                   data_layout=self.data_layout,
                                   parallel_strategy=self.parallel_strategy)
        else:
            return y


class ConvolutionModule(Module):
    """Basic block for convolutional neural networks.

    Applies a convolution and a nonlinear activation function.

    """

    global_count = 0  # Static counter, used for default names

    def __init__(self, num_dims,
                 out_channels, kernel_size, has_vectors=False,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 weights=[], activation=None, name=None, transpose=False,
                 parallel_strategy={}):
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
                and the bias. If weights are not provided, the kernel
                will be initialized with He normal initialization and
                the bias with zeros.
            name (str): Default name is in the form 'convmodule<index>'.
            transpose (bool): If true call deconvolution (or convolution
                         transpose)
            parallel_strategy dict): Data partitioning scheme.

        """
        super().__init__()
        ConvolutionModule.global_count += 1
        self.instance = 0
        self.num_dims = num_dims
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.has_vectors = has_vectors
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.weights = list(make_iterable(weights))
        self.name = (name
                     if name
                     else 'convmodule{0}'.format(ConvolutionModule.global_count))
        self.transpose = transpose
        self.parallel_strategy = parallel_strategy

        # Initialize weights
        # Note: If weights are not provided, kernel weights are
        # initialized with He normal scheme and bias weights are
        # initialized with zeros.
        self.weights = list(make_iterable(weights))
        if len(self.weights) > 2:
            raise ValueError('`ConvolutionModule` has '
                             'at most two weights, '
                             'but got {0}'.format(len(self.weights)))
        if len(self.weights) == 0:
            self.weights.append(
                lbann.Weights(initializer=lbann.HeNormalInitializer(),
                              name=self.name+'_kernel'))
        if len(self.weights) == 1:
            self.weights.append(
                lbann.Weights(initializer=lbann.ConstantInitializer(value=0.0),
                              name=self.name+'_bias'))

        # Initialize activation layer
        self.activation = None
        if activation:
            if isinstance(activation, type):
                self.activation = activation
            else:
                self.activation = type(activation)
            if not issubclass(self.activation, lbann.Layer):
                raise ValueError('activation must be a layer')

    def forward(self, x):
        self.instance += 1
        name = '{0}_instance{1}'.format(self.name, self.instance)

        convtype = ('_deconv' if self.transpose else '_conv')
        kwargs = {}
        kwargs['weights'] = self.weights

        kwargs['name'] = (name+convtype if self.activation else name)
        kwargs['num_dims'] = self.num_dims
        kwargs['num_output_channels'] = self.out_channels
        kwargs['has_bias'] = self.bias
        kwargs['num_groups'] = self.groups
        kwargs['parallel_strategy'] = self.parallel_strategy
        kwargs['has_vectors'] = self.has_vectors

        if (self.has_vectors):
          kwargs['conv_dims'] = self.kernel_size
          kwargs['conv_pads'] = self.padding
          kwargs['conv_dilations'] = self.dilation
          kwargs['conv_strides'] = self.stride
        else:
          kwargs['conv_dims_i'] = self.kernel_size
          kwargs['conv_pads_i'] = self.padding
          kwargs['conv_dilations_i'] = self.dilation
          kwargs['conv_strides_i'] = self.stride


        if(self.transpose):
          y = lbann.Deconvolution(x,**kwargs)
        else:
          y = lbann.Convolution(x,**kwargs)
        if self.activation:
            return self.activation(y, name=name+'_activation')
        else:
            return y

class Convolution2dModule(ConvolutionModule):
    """Basic block for 2D convolutional neural networks.

    Applies a convolution and a nonlinear activation function.
    This is a wrapper class for ConvolutionModule.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(2, *args, **kwargs)

class Convolution3dModule(ConvolutionModule):
    """Basic block for 3D convolutional neural networks.

    Applies a convolution and a nonlinear activation function.
    This is a wrapper class for ConvolutionModule.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(3, *args, **kwargs)
