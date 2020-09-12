import functools
import operator
import os
import os.path
import sys
import numpy as np
import scipy.special

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
np.random.seed(20200909)
_num_samples = 15
_sequence_length = 5
_input_size = 13
_hidden_size = 7
_sample_size = _sequence_length*_input_size + _hidden_size
_samples = np.random.normal(size=(_num_samples,_sample_size)).astype(np.float32)

# Sample access functions
def get_sample(index):
    return _samples[index,:]
def num_samples():
    return _num_samples
def sample_dims():
    return (_sample_size,)

# ==============================================
# NumPy implementation
# ==============================================

def numpy_gru(x, h, ih_matrix, hh_matrix, ih_bias, hh_bias):

    # Cast inputs to float64
    if x.dtype is not np.float64:
        x = x.astype(np.float64)
    if h.dtype is not np.float64:
        h = h.astype(np.float64)
    if ih_matrix.dtype is not np.float64:
        ih_matrix = ih_matrix.astype(np.float64)
    if hh_matrix.dtype is not np.float64:
        hh_matrix = hh_matrix.astype(np.float64)
    if ih_bias.dtype is not np.float64:
        ih_bias = ih_bias.astype(np.float64)
    if hh_bias.dtype is not np.float64:
        hh_bias = hh_bias.astype(np.float64)

    # Dimensions
    sequence_length, input_size = x.shape
    hidden_size = h.shape[0]

    # Unroll GRU
    y = []
    for t in range(sequence_length):
        ih = np.matmul(ih_matrix, x[t]) + ih_bias
        hh = np.matmul(hh_matrix, h) + hh_bias
        r = scipy.special.expit(ih[:hidden_size] + hh[:hidden_size])
        z = scipy.special.expit(ih[hidden_size:2*hidden_size] + hh[hidden_size:2*hidden_size])
        n = np.tanh(ih[2*hidden_size:] + r*hh[2*hidden_size:])
        h = (1-z)*n + z*h
        y.append(h)
    return np.stack(y)

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
    optimizer = lbann.SGD()
    return trainer, model, data_reader, optimizer

def construct_model(lbann):
    """Construct LBANN model.

    Args:
        lbann (module): Module for LBANN Python frontend

    """

    # Input data
    # Note: Sum with a weights layer so that gradient checking will
    # verify that error signals are correct.
    x_weights = lbann.Weights(initializer=lbann.ConstantInitializer(value=0.0),
                              name='input')
    h_weights = lbann.Weights(initializer=lbann.ConstantInitializer(value=0.0),
                              name='inital_hidden')
    input_ = lbann.Identity(lbann.Input())
    input_slice = lbann.Slice(
        input_,
        slice_points=tools.str_list([0, _sequence_length*_input_size, _sample_size]),
    )
    x = lbann.Reshape(input_slice, dims=tools.str_list([_sequence_length,_input_size]))
    x = lbann.Sum(x, lbann.WeightsLayer(weights=x_weights, hint_layer=x))
    h = lbann.Reshape(input_slice, dims=tools.str_list([_hidden_size]),)
    h = lbann.Sum(h, lbann.WeightsLayer(weights=h_weights, hint_layer=h))
    x_lbann = x
    h_lbann = h

    # Objects for LBANN model
    obj = []
    metrics = []
    callbacks = []

    # ------------------------------------------
    # 1-layer, uni-directional GRU
    # ------------------------------------------

    # Weights
    ih_matrix = np.random.normal(size=(3*_hidden_size,_input_size)).astype(np.float32)
    hh_matrix = np.random.normal(size=(3*_hidden_size,_hidden_size)).astype(np.float32)
    ih_bias = np.random.normal(size=(3*_hidden_size,)).astype(np.float32)
    hh_bias = np.random.normal(size=(3*_hidden_size,)).astype(np.float32)
    ih_matrix_weights = lbann.Weights(
        initializer=lbann.ValueInitializer(
            values=tools.str_list(np.nditer(ih_matrix, order='F'))))
    hh_matrix_weights = lbann.Weights(
        initializer=lbann.ValueInitializer(
            values=tools.str_list(np.nditer(hh_matrix, order='F'))))
    ih_bias_weights = lbann.Weights(
        initializer=lbann.ValueInitializer(
            values=tools.str_list(np.nditer(ih_bias))))
    hh_bias_weights = lbann.Weights(
        initializer=lbann.ValueInitializer(
            values=tools.str_list(np.nditer(hh_bias))))

    # LBANN implementation
    x = x_lbann
    h = h_lbann
    y = lbann.GRU(
        x,
        h,
        hidden_size=_hidden_size,
        weights=[ih_matrix_weights,hh_matrix_weights,ih_bias_weights,hh_bias_weights],
    )
    z = lbann.L2Norm2(y)
    obj.append(z)
    metrics.append(lbann.Metric(z, name='1-layer, unidirectional'))

    # NumPy implementation
    vals = []
    for i in range(num_samples()):
        input_ = get_sample(i).astype(np.float64)
        x = input_[:_sequence_length*_input_size].reshape((_sequence_length,_input_size))
        h = input_[_sequence_length*_input_size:]
        y = numpy_gru(x, h, ih_matrix, hh_matrix, ih_bias, hh_bias)
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
# for test in tools.create_tests(setup_experiment, __file__):
#     globals()[test.__name__] = test
