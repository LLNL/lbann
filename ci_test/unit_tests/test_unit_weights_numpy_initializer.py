import functools
import operator
import os
import os.path
import sys
import numpy as np

# Bamboo utilities
current_file = os.path.realpath(__file__)
current_dir = os.path.dirname(current_file)
weights_dir = os.path.join(current_dir, 'temp')
sys.path.insert(0, os.path.join(os.path.dirname(current_dir), 'common_python'))
import tools
os.makedirs(weights_dir, exist_ok=True)

# ==============================================
# Objects for Python data reader
# ==============================================
# Note: The Python data reader imports this file as a module and calls
# the functions below to ingest data.

# Data
np.random.seed(20210826)
_num_samples = 11
_sample_dims = (4,3,2)
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

def setup_experiment(lbann, weekly):
    """Construct LBANN experiment.

    Args:
        lbann (module): Module for LBANN Python frontend

    """
    mini_batch_size = num_samples()
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
    x = lbann.Input(data_field='samples')
    x_lbann = x

    # Objects for LBANN model
    metrics = []
    callbacks = []

    # ------------------------------------------
    # Data-parallel weights layer
    # ------------------------------------------
    # Note: Weights are stored in one column of (STAR,STAR)
    # distributed matrix

    # Weights
    weights_values = np.random.normal(size=_sample_dims).astype(np.float32)
    weights_file = os.path.join(weights_dir, 'dataparallel_weights.npy')
    np.save(weights_file, weights_values)

    # LBANN implementation
    x = lbann.Reshape(x_lbann, dims=_sample_dims)
    weights = lbann.Weights(
        initializer=lbann.NumpyInitializer(file=weights_file),
    )
    weights = lbann.WeightsLayer(
        weights=weights,
        dims=_sample_dims,
    )
    y = lbann.Multiply(x, weights)
    z = lbann.L2Norm2(y)
    metrics.append(lbann.Metric(z, name='data-parallel weights layer'))

    # NumPy implementation
    vals = []
    for i in range(num_samples()):
        x = get_sample(i).reshape(_sample_dims).astype(np.float64)
        y = x * weights_values
        z = tools.numpy_l2norm2(y)
        vals.append(z)
    val = np.mean(vals)
    tol = 8 * val * np.finfo(np.float32).eps
    callbacks.append(lbann.CallbackCheckMetric(
        metric=metrics[-1].name,
        lower_bound=val-tol,
        upper_bound=val+tol,
        error_on_failure=True,
        execution_modes='test'))

    # ------------------------------------------
    # Data-parallel FC layer
    # ------------------------------------------
    # Note: Weights are stored in (STAR,STAR) distributed matrix

    # Weights
    output_size = 7
    linearity = np.random.normal(size=(output_size, _sample_size)).astype(np.float32)
    linearity = linearity.astype(np.float64)
    bias = np.random.normal(size=output_size).astype(np.float32)
    linearity_file = os.path.join(weights_dir, 'dataparallel_fc_linearity.npy')
    bias_file = os.path.join(weights_dir, 'dataparallel_fc_bias.npy')
    np.save(linearity_file, linearity)
    np.save(bias_file, bias)

    # LBANN implementation
    x = x_lbann
    linearity_weights \
        = lbann.Weights(initializer=lbann.NumpyInitializer(file=linearity_file))
    bias_weights \
        = lbann.Weights(initializer=lbann.NumpyInitializer(file=bias_file))
    y = lbann.FullyConnected(
        x,
        weights=(linearity_weights, bias_weights),
        data_layout='data_parallel',
        num_neurons=output_size,
        has_bias=True,
        transpose=False)
    z = lbann.L2Norm2(y)
    metrics.append(lbann.Metric(z, name='data-parallel FC layer'))

    # NumPy implementation
    vals = []
    for i in range(num_samples()):
        x = get_sample(i).astype(np.float64)
        y = np.matmul(linearity, x) + bias
        z = tools.numpy_l2norm2(y)
        vals.append(z)
    val = np.mean(vals)
    tol = 8 * val * np.finfo(np.float32).eps
    callbacks.append(lbann.CallbackCheckMetric(
        metric=metrics[-1].name,
        lower_bound=val-tol,
        upper_bound=val+tol,
        error_on_failure=True,
        execution_modes='test'))

    # ------------------------------------------
    # Model-parallel FC layer
    # ------------------------------------------
    # Note: Weights are stored in (MC,MR) distributed matrix

    # Weights
    output_size = 9
    linearity = np.random.normal(size=(output_size, _sample_size)).astype(np.float32)
    bias = np.random.normal(size=output_size).astype(np.float32)
    bias = bias.astype(np.float64)
    linearity_file = os.path.join(weights_dir, 'modelparallel_fc_linearity.npy')
    bias_file = os.path.join(weights_dir, 'modelparallel_fc_bias.npy')
    np.save(linearity_file, linearity)
    np.save(bias_file, bias)

    # LBANN implementation
    x = x_lbann
    linearity_weights \
        = lbann.Weights(initializer=lbann.NumpyInitializer(file=linearity_file))
    bias_weights \
        = lbann.Weights(initializer=lbann.NumpyInitializer(file=bias_file))
    y = lbann.FullyConnected(
        x,
        weights=(linearity_weights, bias_weights),
        data_layout='model_parallel',
        num_neurons=output_size,
        has_bias=True,
        transpose=False)
    z = lbann.L2Norm2(y)
    metrics.append(lbann.Metric(z, name='model-parallel FC layer'))

    # NumPy implementation
    vals = []
    for i in range(num_samples()):
        x = get_sample(i).astype(np.float64)
        y = np.matmul(linearity, x) + bias
        z = tools.numpy_l2norm2(y)
        vals.append(z)
    val = np.mean(vals)
    tol = 8 * val * np.finfo(np.float32).eps
    callbacks.append(lbann.CallbackCheckMetric(
        metric=metrics[-1].name,
        lower_bound=val-tol,
        upper_bound=val+tol,
        error_on_failure=True,
        execution_modes='test'))

    # ------------------------------------------
    # Construct model
    # ------------------------------------------

    num_epochs = 0
    return lbann.Model(num_epochs,
                       layers=lbann.traverse_layer_graph(x_lbann),
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
