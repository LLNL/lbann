"""Neural network modules.

These are a convenience for common layer patterns that are often the
basic building blocks for larger models.

"""

from collections.abc import Iterable
import warnings
from math import sqrt
import lbann
from lbann.util import make_iterable

def _str_list(l):
    """Convert an iterable object to a space-separated string."""
    return ' '.join(str(i) for i in make_iterable(l))

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
                fully-connected layer. There are at most two: the
                matrix and the bias. If weights are not provided, the
                matrix will be initialized with He normal
                initialization and the bias with zeros.
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
        y = lbann.FullyConnected(x,
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

class ConvolutionModule(Module):
    """Basic block for convolutional neural networks.

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
                and the bias. If weights are not provided, the kernel
                will be initialized with He normal initialization and
                the bias with zeros.
            name (str): Default name is in the form 'convmodule<index>'.

        """
        super().__init__()
        ConvolutionModule.global_count += 1
        self.instance = 0
        self.num_dims = num_dims
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.weights = list(make_iterable(weights))
        self.name = (name
                     if name
                     else 'convmodule{0}'.format(ConvolutionModule.global_count))

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
        y = lbann.Convolution(x,
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

class LSTMCell(Module):
    """Long short-term memory cell."""

    global_count = 0  # Static counter, used for default names

    def __init__(self, size, bias = True,
                 weights=[], name=None, data_layout='data_parallel'):
        """Initialize LSTM cell.

        Args:
            size (int): Size of output tensor.
            bias (bool): Whether to apply biases after linearity.
            weights (`Weights` or iterator of `Weights`): Weights in
                fully-connected layer. There are at most two - a
                matrix ((4*size) x (input_size+size) dimensions) and a
                bias (4*size entries). If weights are not provided,
                the matrix and bias will be initialized in a similar
                manner as PyTorch (uniform random values from
                [-1/sqrt(size), 1/sqrt(size)]).
            name (str): Default name is in the form 'lstmcell<index>'.
            data_layout (str): Data layout.

        """
        super().__init__()
        LSTMCell.global_count += 1
        self.step = 0
        self.size = size
        self.name = (name
                     if name
                     else 'lstmcell{0}'.format(LSTMCell.global_count))
        self.data_layout = data_layout

        # Initial state
        self.last_output = lbann.Constant(value=0.0, num_neurons=str(size),
                                          name=self.name + '_init_output',
                                          data_layout=self.data_layout)
        self.last_cell = lbann.Constant(value=0.0, num_neurons=str(size),
                                        name=self.name + '_init_cell',
                                        data_layout=self.data_layout)

        # Weights
        self.weights = list(make_iterable(weights))
        if len(self.weights) > 2:
            raise ValueError('`LSTMCell` has at most two weights, '
                             'but got {0}'.format(len(self.weights)))
        if len(self.weights) == 0:
            self.weights.append(
                lbann.Weights(initializer=lbann.UniformInitializer(min=-1/sqrt(self.size),
                                                                   max=-1/sqrt(self.size)),
                              name=self.name+'_matrix'))
        if len(self.weights) == 1:
            self.weights.append(
                lbann.Weights(initializer=lbann.UniformInitializer(min=-1/sqrt(self.size),
                                                                   max=-1/sqrt(self.size)),
                           name=self.name+'_bias'))

        # Linearity
        self.fc = FullyConnectedModule(4*size, bias=bias,
                                       weights=self.weights,
                                       name=self.name + '_fc',
                                       data_layout=self.data_layout)

    def forward(self, x):
        """Perform LSTM step.

        State from previous steps is used to compute output.

        """
        self.step += 1
        name = '{0}_step{1}'.format(self.name, self.step)

        # Apply linearity
        input_concat = lbann.Concatenation([x, self.last_output],
                                           name=name + '_input',
                                           data_layout=self.data_layout)
        fc = self.fc(input_concat)

        # Get gates and cell update
        slice = lbann.Slice(fc,
                            slice_points=_str_list([0, self.size, 4*self.size]),
                            name=name + '_fc_slice',
                            data_layout=self.data_layout)
        cell_update = lbann.Tanh(slice,
                                 name=name + '_cell_update',
                                 data_layout=self.data_layout)
        sigmoid = lbann.Sigmoid(slice,
                                name=name + '_sigmoid',
                                data_layout=self.data_layout)
        slice = lbann.Slice(sigmoid,
                            slice_points=_str_list([0, self.size, 2*self.size, 3*self.size]),
                            name=name + '_sigmoid_slice',
                            data_layout=self.data_layout)
        f = lbann.Identity(slice, name=name + '_forget_gate',
                           data_layout=self.data_layout)
        i = lbann.Identity(slice, name=name + '_input_gate',
                           data_layout=self.data_layout)
        o = lbann.Identity(slice, name=name + '_output_gate',
                           data_layout=self.data_layout)

        # Cell state
        cell_forget = lbann.Multiply([f, self.last_cell],
                                     name=name + '_cell_forget',
                                     data_layout=self.data_layout)
        cell_input = lbann.Multiply([i, cell_update],
                                    name=name + '_cell_input',
                                    data_layout=self.data_layout)
        cell = lbann.Add([cell_forget, cell_input], name=name + '_cell',
                         data_layout=self.data_layout)

        # Output
        cell_act = lbann.Tanh(cell, name=name + '_cell_activation',
                              data_layout=self.data_layout)
        output = lbann.Multiply([o, cell_act], name=name,
                                data_layout=self.data_layout)

        # Update state and return output
        self.last_cell = cell
        self.last_output = output
        return output
