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

    A module is a pattern of layers that can be added to a layer
    graph, possibly multiple times. The pattern typically takes a set
    of input layers and obtains a set of output layers.

    """

    def __init__(self):
        pass

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

        # Weights
        self.weights = list(make_iterable(weights))
        if len(self.weights) > 2:
            raise ValueError('`LSTMCell` has at most two weights, '
                             'but got {0}'.format(len(self.weights)))
        if len(self.weights) == 0:
            self.weights.append(
                lbann.Weights(initializer=lbann.UniformInitializer(min=-1/sqrt(self.size),
                                                                   max=1/sqrt(self.size)),
                              name=self.name+'_matrix'))
        if len(self.weights) == 1:
            self.weights.append(
                lbann.Weights(initializer=lbann.UniformInitializer(min=-1/sqrt(self.size),
                                                                   max=1/sqrt(self.size)),
                              name=self.name+'_bias'))

        # Linearity
        self.fc = FullyConnectedModule(4*size, bias=bias,
                                       weights=self.weights,
                                       name=self.name + '_fc',
                                       data_layout=self.data_layout)

    def forward(self, x, prev_state):
        """Apply LSTM step.

        Args:
            x (Layer): Input.
            prev_state (tuple with two `Layer`s): State from previous
                LSTM step. Comprised of LSTM output and cell state.

        Returns:
            (Layer, (Layer, Layer)): The output and state (the output
                and cell state). The state can be passed directly into
                the next LSTM step.

        """
        self.step += 1
        name = '{0}_step{1}'.format(self.name, self.step)

        # Get output and cell state from previous step
        prev_output, prev_cell = prev_state

        # Apply linearity
        input_concat = lbann.Concatenation([x, prev_output],
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
        cell_forget = lbann.Multiply([f, prev_cell],
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

        # Return output and state
        return output, (output, cell)

class GRU(Module):
    """Gated-recurrent unit.
       Implementation mostly taken from:
       https://pytorch.org/docs/stable/nn.html#gru"""

    global_count = 0  # Static counter, used for default names

    def __init__(self, size, bias = True,
                 weights=[], name=None, data_layout='data_parallel'):
        """Initialize GRU cell.

        Args:
            size (int): Size of output tensor.
            bias (bool): Whether to apply biases after linearity.
            weights (`Weights` or iterator of `Weights`): Weights in
                fully-connected layer. There are at most four - two
                matrices ((3*size) x (input_size) and (3*size) x (size) dimensions) each and two
                biases (3*size entries) each. If weights are not provided,
                the matrix and bias will be initialized in a similar
                manner as PyTorch (uniform random values from
                [-1/sqrt(size), 1/sqrt(size)]).
            name (str): Default name is in the form 'gru<index>'.
            data_layout (str): Data layout.

        """
        super().__init__()
        GRU.global_count += 1
        self.step = 0
        self.size = size
        self.name = (name
                     if name
                     else 'gru{0}'.format(GRU.global_count))
        self.data_layout = data_layout

        # Weights
        self.weights = list(make_iterable(weights))
        if len(self.weights) > 4:
            raise ValueError('`GRU` has at most 4 weights, '
                             'but got {0}'.format(len(self.weights)))
        ##@todo: use loop
        if len(self.weights) == 0:
            self.weights.append(
                lbann.Weights(initializer=lbann.UniformInitializer(min=-1/sqrt(self.size),
                                                                   max=1/sqrt(self.size)),
                              name=self.name+'_ih_matrix'))
        if len(self.weights) == 1:
            self.weights.append(
                lbann.Weights(initializer=lbann.UniformInitializer(min=-1/sqrt(self.size),
                                                                   max=1/sqrt(self.size)),
                              name=self.name+'_ih_bias'))
        if len(self.weights) == 2:
            self.weights.append(
                lbann.Weights(initializer=lbann.UniformInitializer(min=-1/sqrt(self.size),
                                                                   max=1/sqrt(self.size)),
                              name=self.name+'_hh_matrix'))
        if len(self.weights) == 3:
            self.weights.append(
                lbann.Weights(initializer=lbann.UniformInitializer(min=-1/sqrt(self.size),
                                                                   max=1/sqrt(self.size)),
                              name=self.name+'_hh_bias'))

        # Linearity
        ####Learnable input-hidden weights
        self.ih_fc = lbann.modules.FullyConnectedModule(3*size, bias=bias,
                                       weights=self.weights[:2],
                                       name=self.name + '_ih_fc',
                                       data_layout=self.data_layout)
        ###Learnable hidden-hidden weights
        self.hh_fc = lbann.modules.FullyConnectedModule(3*size, bias=bias,
                                       weights=self.weights[2:],
                                       name=self.name + '_hh_fc',
                                       data_layout=self.data_layout)

    def forward(self, x, prev_state):
        """Apply GRU step.

        Args:
            x (Layer): Input.
            prev_state: State from previous GRU step.

        Returns:
            (Layer, Layer): The output (out)  and state (hn). 
                          The state can be passed directly into
                           the next GRU step.

        """
        self.step += 1
        name = '{0}_step{1}'.format(self.name, self.step)


        fc1 = self.ih_fc(x)   #input_fc
        fc2 = self.hh_fc(prev_state)  #hidden_fc


        # Get gates and cell update
        fc1_slice = lbann.Slice(fc1,
                            slice_points=_str_list([0, self.size, 2*self.size, 3*self.size]),
                            name=name + '_fc1_slice',
                            data_layout=self.data_layout)
        Wir_x = lbann.Identity(fc1_slice, name=name + '_Wrx',
                           data_layout=self.data_layout)
        Wiz_x = lbann.Identity(fc1_slice, name=name + '_Wzx',
                           data_layout=self.data_layout)
        Win_x = lbann.Identity(fc1_slice, name=name + '_Wnx',
                           data_layout=self.data_layout)

        fc2_slice = lbann.Slice(fc2,
                            slice_points=_str_list([0, self.size, 2*self.size, 3*self.size]),
                            name=name + '_fc2_slice',
                            data_layout=self.data_layout)
        Whr_prev = lbann.Identity(fc2_slice, name=name + '_Wrh',
                           data_layout=self.data_layout)
        Whz_prev = lbann.Identity(fc2_slice, name=name + '_Wzh',
                           data_layout=self.data_layout)
        Whn_prev = lbann.Identity(fc2_slice, name=name + '_Wnh',
                           data_layout=self.data_layout)
        
        rt = lbann.Sigmoid(lbann.Add([Wir_x,Whr_prev], data_layout=self.data_layout), name=name + '_reset_gate',
                           data_layout=self.data_layout)

        zt = lbann.Sigmoid(lbann.Add([Wiz_x,Whz_prev], data_layout=self.data_layout), name=name + '_update_gate',
                           data_layout=self.data_layout)
        
        nt = lbann.Tanh(lbann.Add([Win_x,
                        lbann.Multiply([rt,Whn_prev], data_layout=self.data_layout)], data_layout=self.data_layout),
                        name=name + '_new_gate', data_layout=self.data_layout)

        ht = lbann.Add([
                       lbann.Multiply([
                             lbann.WeightedSum([
                                 lbann.Constant(value=1.0, hint_layer=zt, data_layout=self.data_layout),
                                 zt],
                                 scaling_factors='1 -1', data_layout=self.data_layout),
                             nt], data_layout=self.data_layout),
                       lbann.Multiply([zt,prev_state], data_layout=self.data_layout)], name=name+ '_output', 
                       data_layout=self.data_layout)
        
        # Return output
        return ht, ht
