import functools
import operator
import os
import os.path
import sys
import numpy as np
import pytest

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

# Data
np.random.seed(20191204)
_num_samples = 17
_sample_size = 60
_samples = np.random.normal(size=(_num_samples,_sample_size), loc=1).astype(np.float32)

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
    mini_batch_size = num_samples() // 2
    trainer = lbann.Trainer(mini_batch_size)
    model = construct_model(lbann)
    data_reader = construct_data_reader(lbann)
    optimizer = lbann.NoOptimizer()
    return trainer, model, data_reader, optimizer

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
    x = lbann.Sum(lbann.Input(),
                  lbann.WeightsLayer(weights=x_weights,
                                     dims=tools.str_list(_sample_size)))
    x_lbann = x

    # Objects for LBANN model
    obj = []
    metrics = []
    callbacks = []

    # --------------------------
    # Concatenate along axis 0
    # --------------------------

    # LBANN implementation
    x = x_lbann
    x = lbann.Reshape(x, dims=tools.str_list([5,3,4]))
    x_slice = lbann.Slice(x, axis=0, slice_points=tools.str_list([0,1,3,5]))
    x1 = lbann.Identity(x_slice)
    x2 = lbann.Identity(x_slice)
    x3 = lbann.Identity(x_slice)
    y = lbann.Concatenation(x3, x2, x1, axis=0)
    z = lbann.L2Norm2(lbann.Multiply(x, y))
    obj.append(z)
    metrics.append(lbann.Metric(z, name='axis0'))

    # NumPy implementation
    vals = []
    for i in range(num_samples()):
        x = get_sample(i).reshape([5,3,4]).astype(np.float64)
        x1 = x[0:1,:,:]
        x2 = x[1:3,:,:]
        x3 = x[3:5,:,:]
        y = np.concatenate((x3, x2, x1), axis=0)
        z = tools.numpy_l2norm2(x*y)
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
    x = x_lbann
    x = lbann.Reshape(x, dims=tools.str_list([3,4,5]))
    x_slice = lbann.Slice(x, axis=1, slice_points=tools.str_list([0,1,3,4]))
    x1 = lbann.Identity(x_slice)
    x2 = lbann.Identity(x_slice)
    x3 = lbann.Identity(x_slice)
    y = lbann.Concatenation(x2, x1, x3, axis=1)
    z = lbann.L2Norm2(lbann.Multiply(x, y))
    obj.append(z)
    metrics.append(lbann.Metric(z, name='axis1'))

    # NumPy implementation
    vals = []
    for i in range(num_samples()):
        x = get_sample(i).reshape([3,4,5]).astype(np.float64)
        x1 = x[:,0:1,:]
        x2 = x[:,1:3,:]
        x3 = x[:,3:4,:]
        y = np.concatenate((x2, x1, x3), axis=1)
        z = tools.numpy_l2norm2(x*y)
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
    x = x_lbann
    x = lbann.Reshape(x, dims=tools.str_list([3,4,5]))
    x_slice = lbann.Slice(x, axis=2, slice_points=tools.str_list([0,1,2,3,5]))
    x1 = lbann.Identity(x_slice)
    x2 = lbann.Identity(x_slice)
    x3 = lbann.Identity(x_slice)
    x4 = lbann.Identity(x_slice)
    y = lbann.Concatenation(x2, x4, x1, x3, axis=2)
    z = lbann.L2Norm2(lbann.Multiply(x, y))
    obj.append(z)
    metrics.append(lbann.Metric(z, name='axis2'))

    # NumPy implementation
    vals = []
    for i in range(num_samples()):
        x = get_sample(i).reshape([3,4,5]).astype(np.float64)
        x1 = x[:,:,0:1]
        x2 = x[:,:,1:2]
        x3 = x[:,:,2:3]
        x4 = x[:,:,3:5]
        y = np.concatenate((x2, x4, x1, x3), axis=2)
        z = tools.numpy_l2norm2(x*y)
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
    # Model-parallel
    # --------------------------

    # LBANN implementation
    x = x_lbann
    x = lbann.Reshape(x, dims=tools.str_list([60]))
    x_slice = lbann.Slice(x, slice_points=tools.str_list([0,22,23,60]))
    x1 = lbann.Identity(x_slice)
    x2 = lbann.Identity(x_slice)
    x3 = lbann.Identity(x_slice)
    y = lbann.Concatenation(x3, x1, x2, data_layout='model_parallel')
    z = lbann.L2Norm2(lbann.Multiply(x, y))
    obj.append(z)
    metrics.append(lbann.Metric(z, name='model-parallel'))

    # NumPy implementation
    vals = []
    for i in range(num_samples()):
        x = get_sample(i).reshape([60]).astype(np.float64)
        x1 = x[0:22]
        x2 = x[22:23]
        x3 = x[23:60]
        y = np.concatenate((x3, x1, x2))
        z = tools.numpy_l2norm2(x*y)
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

    callbacks.append(lbann.CallbackCheckGradients(error_on_failure=False))

    # --------------------------
    # Construct model
    # --------------------------

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
for test in tools.create_tests(setup_experiment, __file__):
    globals()[test.__name__] = test
