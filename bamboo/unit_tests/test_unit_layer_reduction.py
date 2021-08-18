import functools
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

# Data
np.random.seed(20210313)
_num_samples = 13
_sample_size = 29
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
    # Note: Slice to separate the last entry in the input tensor. Sum
    # with a weights layer so that gradient checking will verify that
    # error signals are correct.
    x = lbann.Identity(lbann.Input(data_field='samples'))
    x = lbann.Slice(x, slice_points=tools.str_list([0,_sample_size-1,_sample_size]))
    x1 = lbann.Identity(x)
    x2 = lbann.Identity(x)
    x1_weights = lbann.Weights(
        optimizer=lbann.SGD(),
        initializer=lbann.ConstantInitializer(value=0.0),
        name='input_weights')
    x1 = lbann.Sum(x1, lbann.WeightsLayer(weights=x1_weights, hint_layer=x1))
    x1_lbann = x1
    x2_lbann = x2

    # Objects for LBANN model
    obj = []
    metrics = []
    callbacks = []

    # ------------------------------------------
    # Compute expected metric values with NumPy
    # ------------------------------------------

    vals_sum = []
    vals_mean = []
    for i in range(num_samples()):
        x = get_sample(i).astype(np.float64)
        x1 = x[:-1]
        x2 = x[-1]
        y_sum = np.sum(x1)
        z_sum = y_sum * x2
        y_mean = np.mean(x1)
        z_mean = y_mean * x2
        vals_sum.append(z_sum)
        vals_mean.append(z_mean)
    val_sum = np.mean(vals_sum)
    val_mean = np.mean(vals_mean)
    tol = 2 * _sample_size * np.finfo(np.float32).eps

    # ------------------------------------------
    # Data-parallel layout, sum reduction
    # ------------------------------------------

    x1 = x1_lbann
    x2 = x2_lbann
    y = lbann.Reduction(x1, mode='sum', data_layout='data_parallel')
    z = lbann.Multiply(y, x2)
    obj.append(z)
    metrics.append(lbann.Metric(z, name='data-parallel layout, sum reduction'))
    callbacks.append(
        lbann.CallbackCheckMetric(
            metric=metrics[-1].name,
            lower_bound=val_sum-tol,
            upper_bound=val_sum+tol,
            error_on_failure=True,
            execution_modes='test',
        )
    )

    # ------------------------------------------
    # Data-parallel layout, mean reduction
    # ------------------------------------------

    x1 = x1_lbann
    x2 = x2_lbann
    y = lbann.Reduction(x1, mode='mean', data_layout='data_parallel')
    z = lbann.Multiply(y, x2)
    obj.append(z)
    metrics.append(lbann.Metric(z, name='data-parallel layout, mean reduction'))
    callbacks.append(
        lbann.CallbackCheckMetric(
            metric=metrics[-1].name,
            lower_bound=val_mean-tol,
            upper_bound=val_mean+tol,
            error_on_failure=True,
            execution_modes='test',
        )
    )

    # ------------------------------------------
    # Model-parallel layout, sum reduction
    # ------------------------------------------

    x1 = x1_lbann
    x2 = x2_lbann
    y = lbann.Reduction(x1, mode='sum', data_layout='model_parallel')
    z = lbann.Multiply(y, x2)
    obj.append(z)
    metrics.append(lbann.Metric(z, name='model-parallel layout, sum reduction'))
    callbacks.append(
        lbann.CallbackCheckMetric(
            metric=metrics[-1].name,
            lower_bound=val_sum-tol,
            upper_bound=val_sum+tol,
            error_on_failure=True,
            execution_modes='test',
        )
    )

    # ------------------------------------------
    # Data-parallel layout, mean reduction
    # ------------------------------------------

    x1 = x1_lbann
    x2 = x2_lbann
    y = lbann.Reduction(x1, mode='mean', data_layout='model_parallel')
    z = lbann.Multiply(y, x2)
    obj.append(z)
    metrics.append(lbann.Metric(z, name='model-parallel layout, mean reduction'))
    callbacks.append(
        lbann.CallbackCheckMetric(
            metric=metrics[-1].name,
            lower_bound=val_mean-tol,
            upper_bound=val_mean+tol,
            error_on_failure=True,
            execution_modes='test',
        )
    )

    # ------------------------------------------
    # Gradient checking
    # ------------------------------------------

    callbacks.append(lbann.CallbackCheckGradients(error_on_failure=True))

    # ------------------------------------------
    # Construct model
    # ------------------------------------------

    num_epochs = 0
    return lbann.Model(num_epochs,
                       layers=x1_lbann,
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
