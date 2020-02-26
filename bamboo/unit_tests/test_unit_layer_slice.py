import functools
import operator
import os
import os.path
import sys
import numpy as np
import pytest

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
np.random.seed(20190708)
_num_samples = 29
_sample_dims = (7,5,3)
_sample_size = functools.reduce(operator.mul, _sample_dims)
_samples = np.random.normal(size=(_num_samples,_sample_size)).astype(np.float32)

# Sample access functions
def get_sample(index):
    return _samples[index,:]
def num_samples():
    return _num_samples
def sample_dims():
    return (_sample_size,)

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

    # LBANN objects
    obj = []
    metrics = []
    callbacks = []

    # --------------------------
    # LBANN input data
    # --------------------------
    # Note: Sum with a weights layer so that gradient checking will
    # verify that error signals are correct.
    w = lbann.Weights(optimizer=lbann.SGD(),
                      initializer=lbann.ConstantInitializer(value=0.0))
    x0 = lbann.WeightsLayer(weights=w,
                            dims=str_list(_sample_dims))
    x1 = lbann.Reshape(lbann.Input(), dims=str_list(_sample_dims))
    x_lbann = lbann.Sum([x0, x1])

    # --------------------------
    # Slice along axis 0
    # --------------------------

    # LBANN implementation
    slice_points = (2, 3, 6, 7)
    x = x_lbann
    x_slice = lbann.Slice(x, axis=0, slice_points=str_list(slice_points))
    y = []
    for _ in range(len(slice_points)-1):
        y.append(lbann.L2Norm2(x_slice))
    z = lbann.Sum(y)
    obj.append(z)
    metrics.append(lbann.Metric(z, name='axis0'))

    # NumPy implementation
    vals = []
    for i in range(num_samples()):
        x = get_sample(i).reshape(_sample_dims)
        y = []
        for j in range(len(slice_points)-1):
            x_slice = x[slice_points[j]:slice_points[j+1],:,:]
            y.append(l2_norm2(x_slice))
        z = sum(y)
        vals.append(z)
    val = np.mean(vals)
    tol = 8 * val * np.finfo(np.float32).eps
    callbacks.append(lbann.CallbackCheckMetric(
        metric=metrics[-1].name,
        lower_bound=val-tol,
        upper_bound=val+tol,
        error_on_failure=True,
        execution_modes='test'))

    # --------------------------
    # Slice along axis 1
    # --------------------------

    # LBANN implementation
    slice_points = (0, 2, 3, 4)
    x = x_lbann
    x_slice = lbann.Slice(x, axis=1, slice_points=str_list(slice_points))
    y = []
    for _ in range(len(slice_points)-1):
        y.append(lbann.L2Norm2(x_slice))
    z = lbann.Sum(y)
    obj.append(z)
    metrics.append(lbann.Metric(z, name='axis1'))

    # NumPy implementation
    vals = []
    for i in range(num_samples()):
        x = get_sample(i).reshape(_sample_dims)
        y = []
        for j in range(len(slice_points)-1):
            x_slice = x[:,slice_points[j]:slice_points[j+1],:]
            y.append(l2_norm2(x_slice))
        z = sum(y)
        vals.append(z)
    val = np.mean(vals)
    tol = 8 * val * np.finfo(np.float32).eps
    callbacks.append(lbann.CallbackCheckMetric(
        metric=metrics[-1].name,
        lower_bound=val-tol,
        upper_bound=val+tol,
        error_on_failure=True,
        execution_modes='test'))

    # --------------------------
    # Slice along axis 2
    # --------------------------

    # LBANN implementation
    slice_points = (0, 1, 2, 3)
    x = x_lbann
    x_slice = lbann.Slice(x, axis=2, slice_points=str_list(slice_points))
    y = []
    for _ in range(len(slice_points)-1):
        y.append(lbann.L2Norm2(x_slice))
    z = lbann.Sum(y)
    obj.append(z)
    metrics.append(lbann.Metric(z, name='axis2'))

    # NumPy implementation
    vals = []
    for i in range(num_samples()):
        x = get_sample(i).reshape(_sample_dims)
        y = []
        for j in range(len(slice_points)-1):
            x_slice = x[:,:,slice_points[j]:slice_points[j+1]]
            y.append(l2_norm2(x_slice))
        z = sum(y)
        vals.append(z)
    val = np.mean(vals)
    tol = 8 * val * np.finfo(np.float32).eps
    callbacks.append(lbann.CallbackCheckMetric(
        metric=metrics[-1].name,
        lower_bound=val-tol,
        upper_bound=val+tol,
        error_on_failure=True,
        execution_modes='test'))

    # --------------------------
    # Gradient checking
    # --------------------------

    callbacks.append(lbann.CallbackCheckGradients(error_on_failure=True))

    # --------------------------
    # Construct model
    # --------------------------

    mini_batch_size = num_samples() // 2
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
