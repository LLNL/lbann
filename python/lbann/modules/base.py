"""Base class for neural network modules.

This also contains modules for fully-connected and convolution layers.

"""
import abc
import lbann
from lbann.util import make_iterable, str_list


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
        self.data_layout = data_layout
        self.parallel_strategy = parallel_strategy

        self.name = (name
                     if name
                     else 'fcmodule{0}'.format(FullyConnectedModule.global_count))

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
        if self.bias and len(self.weights) == 1:
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

class ChannelwiseFullyConnectedModule(Module):
  """Basic block for channelwise fully-connected neural networks.

    Applies a dense linearity channelwise and a nonlinear activation function.

  """

  global_count = 0

  def __init__(self,
               size,
               bias=False,
               weights=[],
               activation=None,
               transpose=False,
               name=None,
               parallel_strategy={}):
    """Initalize channelwise fully connected module

    Args:
        size (int or list): Dimension of the output tensor
        bias (bool): Whether to apply bias after linearity.
        transpose (bool): Whether to apply transpose of weights
                matrix.
        weights (`Weights` or iterator of `Weights`): Weights in
                fully-connected layer. There are at most two: the
                matrix and the bias. If weights are not provided, the
                matrix will be initialized with He normal
                initialization and the bias with zeros.
        activation (type): Layer class for activation function.
        name (str): Default name is in the form 'channelwisefc<index>'.
        parallel_strategy (dict): Data partitioning scheme.
    """
    super().__init__()
    ChannelwiseFullyConnectedModule.global_count += 1
    self.instance = 0
    self.size = size
    self.bias = bias
    self.transpose = transpose
    self.parallel_strategy = parallel_strategy
    self.name = (name
                 if name
                 else 'channelwisefc{0}'.format(ChannelwiseFullyConnectedModule.global_count))
    self.data_layout = 'data_parallel'

    self.weights = list(make_iterable(weights))
    if len(self.weights) > 2:
        raise ValueError('`FullyConnectedModule` has '
                         'at most two weights, '
                         'but got {0}'.format(len(self.weights)))
    if len(self.weights) == 0:
        self.weights.append(
            lbann.Weights(initializer=lbann.HeNormalInitializer(),
                          name=self.name+'_matrix'))
    if self.bias and len(self.weights) == 1:
        self.weights.append(
            lbann.Weights(initializer=lbann.ConstantInitializer(value=0.0),
                          name=self.name+'_bias'))
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
    y = lbann.ChannelwiseFullyConnected(x,
                                        weights=self.weights,
                                        name=(name+'_fc' if self.activation else name),
                                        data_layout=self.data_layout,
                                        output_channel_dims=self.size,
                                        bias=self.bias,
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

    def __init__(self,
                 num_dims,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 weights=[],
                 activation=None,
                 name=None,
                 transpose=False,
                 parallel_strategy={}):
        """Initialize convolution module.

        Args:
            num_dims (int): Number of dimensions.
            out_channels (int): Number of output channels, i.e. number
                of filters.
            kernel_size (int) or (list): Size of convolution kernel. Either an int for square kernel or list of size num_dims.
            has_vector (bool): If true then call with non-square kernel
              padding, stride, dilation, and padding
            stride (int) or (list): Convolution stride. Either an int for square kernel or list of size num_dims.
            padding (int) or (list): Convolution padding. Either an int for square kernel or list of size num_dims.
            dilation (int) or (list): Convolution dilation. Either an int for square kernel or list of size num_dims.
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
        self.name = (name
                     if name
                     else 'convmodule{0}'.format(ConvolutionModule.global_count))

        self.instance = 0
        self.num_dims = num_dims
        self.out_channels = out_channels

        self.kernel_dims = list(make_iterable(kernel_size))

        if (len(self.kernel_dims)) == 1:
          self.kernel_dims = self.kernel_dims * self.num_dims
        elif (len(self.kernel_dims)) != self.num_dims:
          raise ValueError("Invalid kernel dimensions passed to {}".format(self.name))

        self.stride = list(make_iterable(stride))

        if (len(self.stride)) == 1:
          self.stride = self.stride * self.num_dims
        elif (len(self.stride)) != self.num_dims:
          raise ValueError("Invalid stride dimensions passed to {}".format(self.name))

        self.padding = list(make_iterable(padding))

        if (len(self.padding)) == 1:
          self.padding = self.padding * self.num_dims
        elif (len(self.stride)) != self.num_dims:
          raise ValueError("Invalid padding dimensions passed to {}".format(self.name))

        self.dilation = list(make_iterable(dilation))

        if (len(self.dilation)) == 1:
          self.dilation = self.dilation * self.num_dims
        elif (len(self.dilation)) != self.num_dims:
          raise ValueError("Invalid dilation dimensions passed to {}".format(self.name))

        self.groups = groups

        self.bias = bias
        self.weights = list(make_iterable(weights))
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
        if self.bias and len(self.weights) == 1:
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
        kwargs['has_vectors'] = True

        kwargs['conv_dims'] = str_list(self.kernel_dims)
        kwargs['conv_pads'] = str_list(self.padding)
        kwargs['conv_dilations'] = str_list(self.dilation)
        kwargs['conv_strides'] = str_list(self.stride)


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
