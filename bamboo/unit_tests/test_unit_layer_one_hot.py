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
np.random.seed(201909113)
one_hot_size = 7
_num_samples = 47
_samples = [np.random.uniform(-1, one_hot_size+1) for _ in range(_num_samples)]

# Sample access functions
def get_sample(index):
    return [_samples[index]]
def num_samples():
    return _num_samples
def sample_dims():
    return (1,)

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
    x_lbann = lbann.Identity(lbann.Input())
    y_numpy = np.random.normal(size=one_hot_size).astype(np.float32)
    y_numpy[:] = 1 ### @todo Remove
    y_lbann = lbann.Weights(
        initializer=lbann.ValueInitializer(
            values=tools.str_list(y_numpy)))
    y_lbann = lbann.WeightsLayer(
        weights=y_lbann,
        dims=tools.str_list([one_hot_size]),
    )

    # Objects for LBANN model
    obj = []
    metrics = []
    callbacks = []

    # ------------------------------------------
    # Compute expected metric values with NumPy
    # ------------------------------------------

    vals = []
    for i in range(num_samples()):
        x = int(np.floor(get_sample(i)[0]))
        y = y_numpy
        z = y[x] if (0 <= x < one_hot_size) else 0
        vals.append(z)
    val = np.mean(vals, dtype=np.float64)
    tol = np.abs(8 * val * np.finfo(np.float32).eps)

    # ------------------------------------------
    # Data-parallel layout
    # ------------------------------------------

    x = x_lbann
    y = y_lbann
    x_onehot = lbann.OneHot(
        x,
        size=one_hot_size,
        data_layout='data_parallel',
    )
    z = lbann.MatMul(
        lbann.Reshape(x_onehot, dims='1 -1'),
        lbann.Reshape(y, dims='1 -1'),
        transpose_b=True,
    )
    obj.append(z)
    metrics.append(lbann.Metric(z, name='data-parallel layout'))
    callbacks.append(
        lbann.CallbackCheckMetric(
            metric=metrics[-1].name,
            lower_bound=val-tol,
            upper_bound=val+tol,
            error_on_failure=True,
            execution_modes='test',
        )
    )

    # ------------------------------------------
    # Model-parallel layout
    # ------------------------------------------

    x = x_lbann
    y = y_lbann
    x_onehot = lbann.OneHot(
        x,
        size=one_hot_size,
        data_layout='model_parallel',
    )
    z = lbann.MatMul(
        lbann.Reshape(x_onehot, dims='1 -1'),
        lbann.Reshape(y, dims='1 -1'),
        transpose_b=True,
    )
    obj.append(z)
    metrics.append(lbann.Metric(z, name='model-parallel layout'))
    callbacks.append(
        lbann.CallbackCheckMetric(
            metric=metrics[-1].name,
            lower_bound=val-tol,
            upper_bound=val+tol,
            error_on_failure=True,
            execution_modes='test',
        )
    )

    # ------------------------------------------
    # Construct model
    # ------------------------------------------

    num_epochs = 0
    return lbann.Model(num_epochs,
                       layers=x_lbann,
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
