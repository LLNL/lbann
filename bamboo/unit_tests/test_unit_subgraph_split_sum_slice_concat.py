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
np.random.seed(20211018)
_num_samples = 16 ### @todo Handle non-nice dataset sizes
_sample_size = 6
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
    mini_batch_size = num_samples() // 2
    mini_batch_size = 8 ### @todo Handle non-nice mini-batch sizes
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
    x = lbann.Sum(lbann.Input(data_field='samples'),
                  lbann.WeightsLayer(weights=x_weights,
                                     dims=tools.str_list(_sample_size)))
    x_lbann = x

    # Objects for LBANN model
    obj = []
    metrics = []
    callbacks = []

    # Helper function to create parallel strategies for sub-grid parallelism
    def make_ps(tag):
        return {'sub_branch_tag':tag, 'enable_subgraph':True}

    # --------------------------
    # Split/sum
    # --------------------------

    # LBANN implementation
    x = lbann.Identity(x_lbann)
    x1 = lbann.Identity(x, parallel_strategy=make_ps(1))
    x2 = lbann.Identity(x, parallel_strategy=make_ps(2))
    y1 = lbann.Square(x1)
    y2 = lbann.Sin(x2)
    y = lbann.Sum(y1, y2)
    z = lbann.Reduction(lbann.Multiply(x_lbann, y))
    obj.append(z)
    metrics.append(lbann.Metric(z, name='split/sum'))

    # NumPy implementation
    vals = []
    for i in range(num_samples()):
        x = get_sample(i).astype(np.float64)
        y = x ** 2 + np.sin(x)
        z = np.sum(x * y)
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
    # Slice/concat
    # --------------------------

    # LBANN implementation
    x = x_lbann
    x_slice = lbann.Slice(x, slice_points=tools.str_list([0,3,6]), parallel_strategy=make_ps(0))
    y1 = lbann.Square(x_slice, parallel_strategy=make_ps(1))
    y2 = lbann.Sin(x_slice, parallel_strategy=make_ps(2))
    y = lbann.Concatenation(y1, y2)
    z = lbann.Reduction(lbann.Multiply(x_lbann, y))
    obj.append(z)
    metrics.append(lbann.Metric(z, name='slice/concat'))

    # NumPy implementation
    vals = []
    for i in range(num_samples()):
        x = get_sample(i).astype(np.float64)
        y1 = x[:3] ** 2
        y2 = np.sin(x[3:])
        y = np.concatenate((y1, y2))
        z = np.sum(x * y)
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
    # Split/concat
    # --------------------------

    # LBANN implementation
    x = lbann.Identity(x_lbann)
    x1 = lbann.Identity(x, parallel_strategy=make_ps(1))
    x2 = lbann.Identity(x, parallel_strategy=make_ps(2))
    y1 = lbann.Square(x1)
    y2 = lbann.Sin(x2)
    y = lbann.Concatenation(y1, y2)
    z = lbann.Multiply(lbann.Tessellate(x_lbann, hint_layer=y), y)
    z = lbann.Reduction(z)
    obj.append(z)
    metrics.append(lbann.Metric(z, name='split/concat'))

    # NumPy implementation
    vals = []
    for i in range(num_samples()):
        x = get_sample(i).astype(np.float64)
        y1 = x ** 2
        y2 = np.sin(x)
        y = np.concatenate((y1, y2))
        z = x.reshape((1,-1)) * y.reshape((2,-1))
        z = np.sum(z)
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
    # Slice/sum
    # --------------------------

    # LBANN implementation
    x = x_lbann
    x_slice = lbann.Slice(x, slice_points=tools.str_list([0,3,6]), parallel_strategy=make_ps(0))
    y1 = lbann.Square(x_slice, parallel_strategy=make_ps(1))
    y2 = lbann.Sin(x_slice, parallel_strategy=make_ps(2))
    y = lbann.Sum(y1, y2)
    z = lbann.Multiply(x_lbann, lbann.Tessellate(y, hint_layer=x_lbann))
    z = lbann.Reduction(z)
    obj.append(z)
    metrics.append(lbann.Metric(z, name='slice/sum'))

    # NumPy implementation
    vals = []
    for i in range(num_samples()):
        x = get_sample(i).astype(np.float64)
        y1 = x[:3] ** 2
        y2 = np.sin(x[3:])
        y = y1 + y2
        z = x.reshape((2,-1)) * y.reshape((1,-1))
        z = np.sum(z)
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
for _test_func in tools.create_tests(setup_experiment, __file__):
    globals()[_test_func.__name__] = _test_func
