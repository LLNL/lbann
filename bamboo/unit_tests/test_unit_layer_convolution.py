import functools
import operator
import os
import os.path
import sys
import numpy as np

# Local files
current_file = os.path.realpath(__file__)
current_dir = os.path.dirname(current_file)
sys.path.insert(0, os.path.join(os.path.dirname(current_dir), 'common_python'))
import tools

# ==============================================
# Objects for Python data reader
# ==============================================
# Note: The Python data reader imports this file as a module and calls
# the functions below to ingest data.

# Data
np.random.seed(20191014)
_samples = np.random.normal(size=(23,2,11,7)).astype(np.float32)
_num_samples = _samples.shape[0]
_sample_dims = _samples.shape[1:]
_sample_size = functools.reduce(operator.mul, _sample_dims)

# Sample access functions
def get_sample(index):
    return _samples[index,:].reshape(-1)
def num_samples():
    return _num_samples
def sample_dims():
    return (_sample_size,)

# ==============================================
# PyTorch implementation
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

    # Convert input data to PyTorch tensors
    import torch
    import torch.nn.functional
    if type(data) is np.ndarray:
        data = torch.from_numpy(data)
    if type(kernel) is np.ndarray:
        kernel = torch.from_numpy(kernel)
    if type(bias) is np.ndarray:
        bias = torch.from_numpy(bias)

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

def setup_experiment(lbann):
    """Construct LBANN experiment.

    Args:
        lbann (module): Module for LBANN Python frontend

    """
    trainer = lbann.Trainer()
    model = construct_model(lbann)
    data_reader = construct_data_reader(lbann)
    optimizer = lbann.NoOptimizer()
    return trainer, model, data_reader, optimizer

def construct_model(lbann):
    """Construct LBANN model.

    Args:
        lbann (module): Module for LBANN Python frontend

    """

    # Convenience function to convert list to a space-separated string
    def str_list(it):
        return ' '.join([str(i) for i in it])

    # Convenience function to compute L2 norm squared with NumPy
    def l2_norm2(x):
        x = x.reshape(-1)
        return np.inner(x, x)

    # Input data
    # Note: Sum with a weights layer so that gradient checking will
    # verify that error signals are correct.
    x_weights = lbann.Weights(optimizer=lbann.SGD(),
                              initializer=lbann.ConstantInitializer(value=0.0),
                              name='input_weights')
    x0 = lbann.WeightsLayer(weights=x_weights,
                            dims=str_list(_sample_dims))
    x1 = lbann.Reshape(lbann.Input(), dims=str_list(_sample_dims))
    x = lbann.Sum([x0, x1])
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
    kernel = np.random.normal(size=kernel_dims).astype(np.float32)
    bias = np.random.normal(size=kernel_dims[0]).astype(np.float32)

    # Apply convolution
    kernel_weights = lbann.Weights(
        optimizer=lbann.SGD(),
        initializer=lbann.ValueInitializer(values=str_list(np.nditer(kernel))),
        name='kernel1'
    )
    bias_weights = lbann.Weights(
        optimizer=lbann.SGD(),
        initializer=lbann.ValueInitializer(values=str_list(np.nditer(bias))),
        name='bias1'
    )
    x = x_lbann
    y = lbann.Convolution(x,
                          weights=(kernel_weights, bias_weights),
                          num_dims=3,
                          num_output_channels=kernel_dims[0],
                          has_vectors=True,
                          conv_dims=str_list(kernel_dims[2:]),
                          conv_strides=str_list(strides),
                          conv_pads=str_list(pads),
                          conv_dilations=str_list(dilations),
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
        z = l2_norm2(y) / _num_samples
        val = z
    except:
        # Precomputed value
        val = 4409.33195701707
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

    mini_batch_size = 11
    num_epochs = 0
    return lbann.Model(mini_batch_size,
                       num_epochs,
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
for test in tools.create_tests(setup_experiment, _test_name):
    globals()[test.__name__] = test
