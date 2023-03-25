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
np.random.seed(20191011)
_num_samples = 31
_input_size = 11
_output_size = 3
_samples = np.random.normal(size=(_num_samples,_input_size)).astype(np.float32)

# Sample access functions
def get_sample(index):
    return _samples[index,:]
def num_samples():
    return _num_samples
def sample_dims():
    return (_input_size,)

# ==============================================
# Setup LBANN experiment
# ==============================================

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
    return trainer, model, data_reader, optimizer, None # Don't request any specific number of nodes

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
                                dims=_input_size),
                  lbann.WeightsLayer(weights=x_weights,
                                     dims=_input_size))
    x_lbann = x

    # Objects for LBANN model
    obj = []
    metrics = []
    callbacks = []

    # ------------------------------------------
    # Compute expected metric values with NumPy
    # ------------------------------------------

    # Weight values
    linearity = np.random.normal(size=(_output_size,_input_size)).astype(np.float32)
    bias = np.random.normal(size=(_output_size,1)).astype(np.float32)

    # With bias
    x = _samples.transpose().astype(np.float64)
    y = np.matmul(linearity.astype(np.float64), x) + bias.astype(np.float64)
    z = tools.numpy_l2norm2(y) / _num_samples
    val_with_bias = z

    # Without bias
    x = _samples.transpose().astype(np.float64)
    y = np.matmul(linearity.astype(np.float64), x)
    z = tools.numpy_l2norm2(y) / _num_samples
    val_without_bias = z

    # ------------------------------------------
    # Data-parallel layout, non-transpose, bias
    # ------------------------------------------

    # LBANN implementation
    linearity_weights = lbann.Weights(
        optimizer=lbann.SGD(),
        initializer=lbann.ValueInitializer(
            values=np.nditer(linearity, order='F')
        )
    )
    bias_weights = lbann.Weights(
        optimizer=lbann.SGD(),
        initializer=lbann.ValueInitializer(
            values=np.nditer(bias)
        )
    )
    x = x_lbann
    y = lbann.FullyConnected(x,
                             weights=(linearity_weights, bias_weights),
                             data_layout='data_parallel',
                             num_neurons=_output_size,
                             has_bias=True,
                             transpose=False)
    z = lbann.L2Norm2(y)
    obj.append(z)
    metrics.append(lbann.Metric(z, name='data-parallel layout, non-transpose, bias'))

    # NumPy implementation
    val = val_with_bias
    tol = 8 * val * np.finfo(np.float32).eps
    callbacks.append(lbann.CallbackCheckMetric(
        metric=metrics[-1].name,
        lower_bound=val-tol,
        upper_bound=val+tol,
        error_on_failure=True,
        execution_modes='test'))

    # ------------------------------------------
    # Model-parallel layout, non-transpose, bias
    # ------------------------------------------

    # LBANN implementation
    linearity_weights = lbann.Weights(
        optimizer=lbann.SGD(),
        initializer=lbann.ValueInitializer(
            values=np.nditer(linearity, order='F')
        )
    )
    bias_weights = lbann.Weights(
        optimizer=lbann.SGD(),
        initializer=lbann.ValueInitializer(
            values=np.nditer(bias)
        )
    )
    x = x_lbann
    y = lbann.FullyConnected(x,
                             weights=(linearity_weights, bias_weights),
                             data_layout='model_parallel',
                             num_neurons=_output_size,
                             has_bias=True,
                             transpose=False)
    z = lbann.L2Norm2(y)
    obj.append(z)
    metrics.append(lbann.Metric(z, name='model-parallel layout, non-transpose, bias'))

    # NumPy implementation
    val = val_with_bias
    tol = 8 * val * np.finfo(np.float32).eps
    callbacks.append(lbann.CallbackCheckMetric(
        metric=metrics[-1].name,
        lower_bound=val-tol,
        upper_bound=val+tol,
        error_on_failure=True,
        execution_modes='test'))

    # ------------------------------------------
    # Data-parallel layout, transpose, no bias
    # ------------------------------------------

    # LBANN implementation
    linearity_weights = lbann.Weights(
        optimizer=lbann.SGD(),
        initializer=lbann.ValueInitializer(
            values=np.nditer(linearity, order='C')
        )
    )
    x = x_lbann
    y = lbann.FullyConnected(x,
                             weights=linearity_weights,
                             data_layout='data_parallel',
                             num_neurons=_output_size,
                             has_bias=False,
                             transpose=True)
    z = lbann.L2Norm2(y)
    obj.append(z)
    metrics.append(lbann.Metric(z, name='data-parallel layout, transpose, no bias'))

    # NumPy implementation
    val = val_without_bias
    tol = 8 * val * np.finfo(np.float32).eps
    callbacks.append(lbann.CallbackCheckMetric(
        metric=metrics[-1].name,
        lower_bound=val-tol,
        upper_bound=val+tol,
        error_on_failure=True,
        execution_modes='test'))

    # ------------------------------------------
    # Model-parallel layout, transpose, no bias
    # ------------------------------------------

    # LBANN implementation
    linearity_weights = lbann.Weights(
        optimizer=lbann.SGD(),
        initializer=lbann.ValueInitializer(
            values=np.nditer(linearity, order='C')
        )
    )
    x = x_lbann
    y = lbann.FullyConnected(x,
                             weights=linearity_weights,
                             data_layout='model_parallel',
                             num_neurons=_output_size,
                             has_bias=False,
                             transpose=True)
    z = lbann.L2Norm2(y)
    obj.append(z)
    metrics.append(lbann.Metric(z, name='data-parallel layout, transpose, no bias'))

    # NumPy implementation
    val = val_without_bias
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
for _test_func in tools.create_tests(setup_experiment, _test_name):
    globals()[_test_func.__name__] = _test_func
