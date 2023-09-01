import functools
import math
import operator
import os
import os.path
import sys
import numpy as np
import pytest
import lbann.contrib.args

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
_num_samples = 8
_sample_dims = [64,16,16]
_sample_dims_3d = [4,16,16,16]
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
        data = data.astype(torch.float64)
    if kernel.dtype is not torch.float64:
        kernel = kernel.astype(torch.float64)
    if bias.dtype is not torch.float64:
        bias = bias.astype(torch.float64)

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
    if not lbann.has_feature('DISTCONV'):
        message = f'{os.path.basename(__file__)} requires DISTCONV'
        print('Skip - ' + message)
        pytest.skip(message)
    
    mini_batch_size = num_samples() // 2
    trainer = lbann.Trainer(mini_batch_size=mini_batch_size)
    model = construct_model(lbann)
    data_reader = construct_data_reader(lbann)
    optimizer = lbann.NoOptimizer()
    return trainer, model, data_reader, optimizer, None # Don't request any specific number of nodes

def create_parallel_strategy(num_height_groups):
    return {"height_groups": num_height_groups}

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
    # Basic 3^n convolution
    # ------------------------------------------
    # 3^n conv, stride=1, pad=1, dilation=1, bias

    num_height_groups = tools.gpus_per_node(lbann)
    if num_height_groups == 0:
        e = 'this test requires GPUs.'
        print('Skip - ' + e)
        pytest.skip(e)

    for num_dims, reference_val in [
            (2, 11913.852660080756),
            (3, 9952.365297083174)]:
        # Convolution settings
        kernel_dims = [5, _sample_dims[0] if num_dims == 2 else _sample_dims_3d[0],] + [3]*num_dims
        strides = [1]*num_dims
        pads = [1]*num_dims
        dilations = [1]*num_dims
        kernel = make_random_array(kernel_dims, 11)

        # Apply convolution
        kernel_weights = lbann.Weights(
            optimizer=lbann.SGD(),
            initializer=lbann.ValueInitializer(values=np.nditer(kernel)),
            name='kernel1_{}d'.format(num_dims)
        )
        x = x_lbann
        if num_dims == 3:
            x = lbann.Reshape(x, dims=_sample_dims_3d)

        y = lbann.Convolution(x,
                              weights=(kernel_weights, ),
                              num_dims=num_dims,
                              out_channels=kernel_dims[0],
                              kernel_size=kernel_dims[2:],
                              stride=strides,
                              padding=pads,
                              dilation=dilations,
                              has_bias=False,
                              parallel_strategy=create_parallel_strategy(
                                  num_height_groups))
        z = lbann.L2Norm2(y)
        obj.append(z)
        metrics.append(lbann.Metric(z, name='basic {}D 3^n convolution'.format(num_dims)))

        # PyTorch implementation
        try:
            x = _samples
            if num_dims == 3:
                x = np.reshape(x, [_num_samples]+_sample_dims_3d)

            y = pytorch_convolution(
                x, kernel,
                stride=strides, padding=pads, dilation=dilations
            )
            z = tools.numpy_l2norm2(y) / _num_samples
            val = z
        except:
            # Precomputed value
            val = reference_val
            # val = 398.6956458317758 # _num_samples=8, 6 channels
            # val = 381.7401227915947 # _num_samples=23, 6 channels
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
for _test_func in tools.create_tests(setup_experiment, _test_name,
                                     environment=lbann.contrib.args.get_distconv_environment(),
                                     time_limit=10):
    globals()[_test_func.__name__] = _test_func
