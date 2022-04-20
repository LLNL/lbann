import functools
import math
import operator
import os
import os.path
import sys
import numpy as np

# Bamboo utilities
current_file = os.path.realpath(__file__)
current_dir = os.path.dirname(current_file)
sys.path.insert(0, os.path.join(os.path.dirname(current_dir), 'common_python'))
import tools

# ==============================================
# Objects for Python data reader
# ==============================================
# Note: The Python data reader imports this file as a module and calls
# the functions below to ingest data.

def make_random_array(shape, seed):
    """Hacked function to generate a random array.

    NumPy's RNG produces different values with different NumPy
    versions. This function is helpful when array values must be
    identical across all runs, e.g. when checking against precomputed
    metric values.

    Args:
        shape (Iterable of int): Array dimensions
        seed (int): Parameter for RNG. Must be non-zero.
    Returns:
        numpy.ndarray: Array of `np.float32`. Values will be in
            [-0.5,0.5).

    """
    size = functools.reduce(operator.mul, shape)
    eps = np.finfo(np.float32).eps
    x = (seed / np.linspace(math.sqrt(eps), 0.1, size)) % 1 - 0.5
    return x.reshape(shape).astype(np.float32)

# Data
_num_samples = 23
_sample_dims = [6,11,7]
_sample_size = functools.reduce(operator.mul, _sample_dims)
_samples = make_random_array([_num_samples] + _sample_dims, 7)

# Sample access functions
def get_sample(index):
    return _samples[index,:].reshape(-1)
def num_samples():
    return _num_samples
def sample_dims():
    return (_sample_size,)

# ==============================================
# PyTorch convolution
# ==============================================

def pytorch_convolution(data,
                        kernel,
                        bias=None,
                        stride=1,
                        padding=0,
                        dilation=1,
                        groups=1):
    """Wrapper around PyTorch convolution.

    Input and output data are NumPy arrays.

    """

    # Convert input data to PyTorch tensors with 64-bit floats
    import torch
    import torch.nn.functional
    if type(data) is np.ndarray:
        data = torch.from_numpy(data)
    if type(kernel) is np.ndarray:
        kernel = torch.from_numpy(kernel)
    if type(bias) is np.ndarray:
        bias = torch.from_numpy(bias)
    if data.dtype is not torch.float64:
        data = data.to(torch.float64)
    if kernel.dtype is not torch.float64:
        kernel = kernel.to(torch.float64)
    if bias is not None and bias.dtype is not torch.float64:
        bias = bias.to(torch.float64)

    # Perform convolution with PyTorch
    output = None
    if len(kernel.shape) == 3:
        output = torch.nn.functional.conv1d(
            data, kernel, bias, stride, padding, dilation, groups
        )
    if len(kernel.shape) == 4:
        output = torch.nn.functional.conv2d(
            data, kernel, bias, stride, padding, dilation, groups
        )
    if len(kernel.shape) == 5:
        output = torch.nn.functional.conv3d(
            data, kernel, bias, stride, padding, dilation, groups
        )
    if output is None:
        raise ValueError('PyTorch only supports 1D, 2D, and 3D convolution')

    # Return output as NumPy array
    return output.numpy()

# ==============================================
# Setup LBANN experiment
# ==============================================

def setup_experiment(lbann, weekly):
    """Construct LBANN experiment.

    Args:
        lbann (module): Module for LBANN Python frontend

    """
    mini_batch_size = num_samples() // 2
    trainer = lbann.Trainer(mini_batch_size)
    model = construct_model(lbann)
    data_reader = construct_data_reader(lbann)
    optimizer = lbann.NoOptimizer()
    return trainer, model, data_reader, optimizer, None # Don't request any specific number of nodes

def construct_model(lbann):
    """Construct LBANN model.

    Args:
        lbann (module): Module for LBANN Python frontend

    """

    # Input data
    # Note: Sum with a weights layer so that gradient checking will
    # verify that error signals are correct.
    x_weights = lbann.Weights(optimizer=lbann.SGD(),
                              initializer=lbann.ConstantInitializer(value=0.0),
                              name='input_weights')
    x = lbann.Sum(lbann.Reshape(lbann.Input(data_field='samples'),
                                dims=_sample_dims),
                  lbann.WeightsLayer(weights=x_weights,
                                     dims=_sample_dims))
    x_lbann = x

    # Objects for LBANN model
    obj = []
    metrics = []
    callbacks = []

    # ------------------------------------------
    # Basic 3x3 convolution
    # ------------------------------------------
    # 3x3 conv, stride=1, pad=1, dilation=1, bias

    # Convolution settings
    kernel_dims = (5, _sample_dims[0], 3, 3)
    strides = (1, 1)
    pads = (1, 1)
    dilations = (1, 1)
    kernel = make_random_array(kernel_dims, 11)
    bias = make_random_array([kernel_dims[0]], 123)

    # Apply convolution
    kernel_weights = lbann.Weights(
        optimizer=lbann.SGD(),
        initializer=lbann.ValueInitializer(values=np.nditer(kernel)),
        name='kernel1'
    )
    bias_weights = lbann.Weights(
        optimizer=lbann.SGD(),
        initializer=lbann.ValueInitializer(values=np.nditer(bias)),
        name='bias1'
    )
    x = x_lbann
    y = lbann.Convolution(x,
                          weights=(kernel_weights, bias_weights),
                          num_dims=2,
                          num_output_channels=kernel_dims[0],
                          has_vectors=True,
                          conv_dims=kernel_dims[2:],
                          conv_strides=strides,
                          conv_pads=pads,
                          conv_dilations=dilations,
                          has_bias=True)
    z = lbann.L2Norm2(y)
    obj.append(z)
    metrics.append(lbann.Metric(z, name='basic 3x3 convolution'))

    # PyTorch implementation
    try:
        x = _samples
        y = pytorch_convolution(
            x, kernel, bias=bias,
            stride=strides, padding=pads, dilation=dilations
        )
        z = tools.numpy_l2norm2(y) / _num_samples
        val = z
    except:
        # Precomputed value
        val = 153.84937996554953
    tol = 8 * val * np.finfo(np.float32).eps
    callbacks.append(lbann.CallbackCheckMetric(
        metric=metrics[-1].name,
        lower_bound=val-tol,
        upper_bound=val+tol,
        error_on_failure=True,
        execution_modes='test'))

    # ------------------------------------------
    # 2x4 strided convolution
    # ------------------------------------------

    # Convolution settings
    kernel_dims = (3, _sample_dims[0], 2, 4)
    strides = (3, 1)
    pads = (3, 0)
    dilations = (1, 1)
    num_groups = 1
    kernel = make_random_array(kernel_dims, 19)

    # Apply convolution
    kernel_weights = lbann.Weights(
        optimizer=lbann.SGD(),
        initializer=lbann.ValueInitializer(values=np.nditer(kernel)),
        name='kernel2'
    )
    x = x_lbann
    y = lbann.Convolution(x,
                          weights=(kernel_weights),
                          num_dims=2,
                          num_output_channels=kernel_dims[0],
                          has_vectors=True,
                          conv_dims=kernel_dims[2:],
                          conv_strides=strides,
                          conv_pads=pads,
                          conv_dilations=dilations,
                          num_groups=num_groups,
                          has_bias=False)
    z = lbann.L2Norm2(y)
    obj.append(z)
    metrics.append(lbann.Metric(z, name='2x4 convolution'))

    # PyTorch implementation
    try:
        x = _samples
        y = pytorch_convolution(
            x, kernel, bias=None,
            stride=strides, padding=pads,
            dilation=dilations, groups=num_groups
        )
        z = tools.numpy_l2norm2(y) / _num_samples
        val = z
    except:
        # Precomputed value
        val = 19.24587403346207
    tol = 8 * val * np.finfo(np.float32).eps
    callbacks.append(lbann.CallbackCheckMetric(
        metric=metrics[-1].name,
        lower_bound=val-tol,
        upper_bound=val+tol,
        error_on_failure=True,
        execution_modes='test'))

    # ------------------------------------------
    # Gradient checking
    # ------------------------------------------

    callbacks.append(lbann.CallbackCheckGradients(error_on_failure=True))

    # ------------------------------------------
    # Construct model
    # ------------------------------------------

    num_epochs = 0
    return lbann.Model(num_epochs,
                       layers=lbann.traverse_layer_graph(x_lbann),
                       objective_function=obj,
                       metrics=metrics,
                       callbacks=callbacks)

def construct_data_reader(lbann):
    """Construct Protobuf message for Python data reader.

    The Python data reader will import the current Python file to
    access the sample access functions.

    Args:
        lbann (module): Module for LBANN Python frontend

    """

    # Note: The training data reader should be removed when
    # https://github.com/LLNL/lbann/issues/1098 is resolved.
    message = lbann.reader_pb2.DataReader()
    message.reader.extend([
        tools.create_python_data_reader(
            lbann,
            current_file,
            'get_sample',
            'num_samples',
            'sample_dims',
            'train'
        )
    ])
    message.reader.extend([
        tools.create_python_data_reader(
            lbann,
            current_file,
            'get_sample',
            'num_samples',
            'sample_dims',
            'test'
        )
    ])
    return message

# ==============================================
# Setup PyTest
# ==============================================

# Create test functions that can interact with PyTest
# Note: Create test name by removing ".py" from file name
_test_name = os.path.splitext(os.path.basename(current_file))[0]
for _test_func in tools.create_tests(setup_experiment, _test_name):
    globals()[_test_func.__name__] = _test_func
