import functools
import numpy as np
import os
import os.path
import sys
import operator
import math
from lbann.modules.transformations import PeriodicPadding3D, PeriodicPadding2D
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
_sample_dims = [6, 11, 7]
_sample_dims_3d = [2, 3, 11, 7]
_sample_size = functools.reduce(operator.mul, _sample_dims)
_samples = make_random_array([_num_samples] + _sample_dims, 7)


# Sample access functions
def get_sample(index):
    return _samples[index, :].reshape(-1)


def num_samples():
    return _num_samples


def sample_dims():
    return (_sample_size,)

# ==============================================
# Periodic Padding
# ==============================================


def periodic_padding_2D(data, padding):
    """
    Args:
      data (np.array) : Input array of shape (B, c, h, w)
      padding (int) : Amount of padding around data
    Returns
      (np.array): Padded atensor with shape
                  (B, c, h+2*padding, h+2*padding)
    """
    _, c, h, w = data.shape
    top_slice = data[:, :, :padding, :]
    bottom_slice = data[:, :, h - padding:, :]
    inter = np.concatenate((bottom_slice, data, top_slice), axis=2)
    left_slice = inter[:, :, :, :padding]
    right_slice = inter[:, :, :, w - padding:]
    return np.concatenate((right_slice, inter, left_slice), axis=3)


def periodic_padding_3D(data, padding):
    """
    Args:
      data (np.array) : Input array of shape (B, c, d, h, w)
      padding (int) : Amount of padding around data
    Returns
      (np.array): Padded atensor with shape
                  (B, c, d+2*padding, h+2*padding, h+2*padding)
    """
    _, c, d, h, w = data.shape
    d_slice_start = data[:, :, :padding, :, :]
    d_slice_end = data[:, :, d - padding:, :, :]
    inter = np.concatenate((d_slice_end, data, d_slice_start), axis=2)
    h_slice_start = inter[:, :, :, :padding, :]
    h_slice_end = inter[:, :, :, h - padding:, :]

    inter = np.concatenate((h_slice_end, inter, h_slice_start), axis=3)

    w_slice_start = inter[:, :, :, :, :padding]
    w_slice_end = inter[:, :, :, :, w - padding:]

    return_val = np.concatenate((w_slice_end, inter, w_slice_start), axis=4)
    return return_val


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
    return trainer, model, data_reader, optimizer, None  # Don't request any specific number of nodes


def construct_model(lbann):
    """Construct LBANN model.

    Args:
        lbann (module): Module for LBANN Python frontend

    """

    # Input data
    # Note: Sum with a weights layer so that gradient checking will
    # verify that error signals are correct.
    x_weights = lbann.Weights(optimizer=lbann.SGD(),
                              initializer=lbann.ConstantInitializer(value=0.0))
    x = lbann.Sum(lbann.Reshape(lbann.Input(data_field='samples'),
                                dims=_sample_dims),
                  lbann.WeightsLayer(weights=x_weights,
                                     dims=_sample_dims))
    x_lbann = x

    # Objects for LBANN model
    obj = []
    metrics = []
    callbacks = []

    x_2D = lbann.Reshape(x_lbann,
                         dims=_sample_dims)
    y = PeriodicPadding2D(x_2D,
                          _sample_dims[1],
                          _sample_dims[2],
                          padding=2)
    z = lbann.L2Norm2(y)
    obj.append(z)
    metrics.append(lbann.Metric(z, name="Padding_2D"))

    x_np = _samples
    y_np = periodic_padding_2D(x_np, padding=2)
    z_np = tools.numpy_l2norm2(y_np) / _num_samples
    tol = 8 * z_np * np.finfo(np.float32).eps

    metric_callback_2d = lbann.CallbackCheckMetric(metric=metrics[-1].name,
                                                   lower_bound=z_np - tol,
                                                   upper_bound=z_np + tol,
                                                   error_on_failure=True,
                                                   execution_modes='test')

    x_3D = lbann.Reshape(x_lbann,
                         dims=_sample_dims_3d)
    y = PeriodicPadding3D(x_3D,
                          _sample_dims_3d[1],
                          _sample_dims_3d[2],
                          _sample_dims_3d[3],
                          padding=1)
    z = lbann.L2Norm2(y)
    obj.append(z)
    metrics.append(lbann.Metric(z, name="Padding_3D"))
    x_np = _samples.reshape([_num_samples] + _sample_dims_3d)
    y_np = periodic_padding_3D(x_np, padding=1)
    z_np = tools.numpy_l2norm2(y_np) / _num_samples

    tol = 8 * z_np * np.finfo(np.float32).eps

    metric_callback_3d = lbann.CallbackCheckMetric(metric=metrics[-1].name,
                                                   lower_bound=z_np - tol,
                                                   upper_bound=z_np + tol,
                                                   error_on_failure=True,
                                                   execution_modes='test')
    metrics.append(lbann.Metric(z, name="Padding_3D"))

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
for _test_func in tools.create_tests(setup_experiment, _test_name, skip_clusters=["catalyst"]):
    globals()[_test_func.__name__] = _test_func
