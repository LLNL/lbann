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
np.random.seed(20200115)
_num_samples = 15
_sample_dims = (5, 2, 7)
_sample_size = functools.reduce(operator.mul, _sample_dims)
_samples = np.random.normal(loc=0.5, size=(_num_samples,
                                           _sample_size)).astype(np.float32)


# Sample access functions
def get_sample(index):
    return _samples[index, :]


def num_samples():
    return _num_samples


def sample_dims():
    return (_sample_size, )


# ==============================================
# NumPy implementation
# ==============================================


def numpy_channelwise_softmax(x, axis=None):
    if x.dtype is not np.float64:
        x = x.astype(np.float64)
    if axis is None:
        axis = tuple(range(1, x.ndim))
    shift = np.max(x, axis=axis, keepdims=True)
    y = np.exp(x - shift)
    return y / np.sum(y, axis=axis, keepdims=True)


# ==============================================
# Setup LBANN experiment
# ==============================================


def setup_experiment(test_case, lbann, weekly):
    """Construct LBANN experiment.

    Args:
        lbann (module): Module for LBANN Python frontend

    """
    mini_batch_size = num_samples() // 2
    trainer = lbann.Trainer(mini_batch_size)
    model = construct_model(lbann, test_case)
    data_reader = construct_data_reader(lbann)
    optimizer = lbann.NoOptimizer()
    return trainer, model, data_reader, optimizer, None  # Don't request any specific number of nodes


def construct_model(lbann, test_case):
    """Construct LBANN model.

    Args:
        lbann (module): Module for LBANN Python frontend
        test_case (str): Name of test case to use

    """

    # Input data
    # Note: Sum with a weights layer so that gradient checking will
    # verify that error signals are correct.
    x_weights = lbann.Weights(optimizer=lbann.SGD(),
                              initializer=lbann.ConstantInitializer(value=0.0),
                              name='input_weights')
    x = lbann.Sum(
        lbann.Reshape(lbann.Input(data_field='samples'), dims=_sample_dims),
        lbann.WeightsLayer(weights=x_weights, dims=_sample_dims))
    x_lbann = x

    # Objects for LBANN model
    obj = []
    metrics = []
    callbacks = []

    # ------------------------------------------
    # Data-parallel layout
    # ------------------------------------------

    # LBANN implementation
    x = x_lbann

    if test_case == 'data-parallel layout':
        y = lbann.ChannelwiseSoftmax(x, data_layout='data_parallel')
        axis = None
    elif test_case == 'single dimension mode':
        y = lbann.ChannelwiseSoftmax(x,
                                     dim=-1,
                                     single_dim_mode=True,
                                     data_layout='data_parallel')
        axis = -1

    z = lbann.L2Norm2(y)
    obj.append(z)
    metrics.append(lbann.Metric(z, name=test_case))

    # NumPy implementation
    vals = []
    for i in range(num_samples()):
        x = get_sample(i).reshape(_sample_dims).astype(np.float64)
        y = numpy_channelwise_softmax(x, axis)
        z = tools.numpy_l2norm2(y)
        vals.append(z)
    val = np.mean(vals)
    tol = 8 * val * np.finfo(np.float32).eps
    callbacks.append(
        lbann.CallbackCheckMetric(metric=metrics[-1].name,
                                  lower_bound=val - tol,
                                  upper_bound=val + tol,
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
        tools.create_python_data_reader(lbann, current_file, 'get_sample',
                                        'num_samples', 'sample_dims', 'train')
    ])
    message.reader.extend([
        tools.create_python_data_reader(lbann, current_file, 'get_sample',
                                        'num_samples', 'sample_dims', 'test')
    ])
    return message


# ==============================================
# Setup PyTest
# ==============================================

# Create test functions that can interact with PyTest
for _test_func in tools.create_tests(
        functools.partial(setup_experiment, 'data-parallel layout'),
        __file__, 'dplayout'):
    globals()[_test_func.__name__] = _test_func
for _test_func in tools.create_tests(
        functools.partial(setup_experiment, 'single dimension mode'),
        __file__, 'singledim'):
    globals()[_test_func.__name__] = _test_func
