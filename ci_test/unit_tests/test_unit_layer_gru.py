import functools
import operator
import os
import os.path
import sys
import numpy as np
import pytest

try:
    import scipy.special
except (ImportError, ModuleNotFoundError):
    pytest.skip('This test requires scipy to run', allow_module_level=True)

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
_sequence_length = 9
_input_size = 5
_num_layers = 2
_sample_size = _sequence_length*_input_size + _num_layers*_input_size
_samples = np.random.uniform(low=-1, high=1, size=(_num_samples,_sample_size))
_samples = _samples.astype(np.float32)

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

def numpy_gru(x, h, w):

    # Cast inputs to float64
    def to_float64_list(a):
        return [a_
                if a_.dtype is np.float64
                else a_.astype(np.float64)
                for a_ in a]
    x = to_float64_list(x)
    h = to_float64_list(h)
    w = to_float64_list(w)

    # Dimensions
    sequence_length = len(x)
    input_size = x[0].size
    num_layers = len(h)
    hidden_size = h[0].size
    assert len(w) == 4*num_layers, 'incorrect number of weights'

    # Unroll GRU
    for i in range(num_layers):
        for j in range(sequence_length):
            ih = np.matmul(w[4*i], x[j]) + w[4*i+2]
            hh = np.matmul(w[4*i+1], h[i]) + w[4*i+3]
            r = scipy.special.expit(ih[:hidden_size] + hh[:hidden_size])
            z = scipy.special.expit(ih[hidden_size:2*hidden_size] + hh[hidden_size:2*hidden_size])
            n = np.tanh(ih[2*hidden_size:] + r*hh[2*hidden_size:])
            h[i] = (1-z)*n + z*h[i]
            x[j] = h[i]
    return np.stack(x)

# ==============================================
# Setup LBANN experiment
# ==============================================

def setup_experiment(lbann, weekly):
    """Construct LBANN experiment.

    Args:
        lbann (module): Module for LBANN Python frontend

    """

    # Skip test when not supported.    
    if not tools.gpus_per_node(lbann):   # CPU system
        if not lbann.has_feature('ONEDNN_CPU'):
            message = f'{os.path.basename(__file__)} requires oneDNN on CPU'
            print('Skip - ' + message)
            pytest.skip(message)
    else:                                # GPU system
        if not (lbann.has_feature('CUDA') and lbann.has_feature('CUDNN')):
            message = f'{os.path.basename(__file__)} requires CUDA and cuDNN on GPU'
            print('Skip - ' + message)
            pytest.skip(message)

    mini_batch_size = num_samples() // 2
    trainer = lbann.Trainer(mini_batch_size)
    model = construct_model(lbann)
    data_reader = construct_data_reader(lbann)
    optimizer = lbann.SGD()
    return trainer, model, data_reader, optimizer, None # Don't request any specific number of nodes

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
    input_ = lbann.Input(data_field='samples')
    input_slice = lbann.Slice(
        input_,
        slice_points=[0, _sequence_length*_input_size, _sample_size],
    )
    x = lbann.Reshape(input_slice, dims=[_sequence_length,_input_size])
    x = lbann.Sum(x, lbann.WeightsLayer(weights=x_weights, hint_layer=x))
    h = lbann.Reshape(input_slice, dims=[_num_layers,_input_size],)
    h = lbann.Sum(h, lbann.WeightsLayer(weights=h_weights, hint_layer=h))
    x_lbann = x
    h_lbann = h

    # Objects for LBANN model
    obj = []
    metrics = []
    callbacks = []

    # ------------------------------------------
    # Multi-layer, unidirectional GRU
    # ------------------------------------------
    # Note: input_size=hidden_size due to a limitation in oneDNN

    # Weights
    rnn_weights_numpy = []
    for i in range(_num_layers):
        ih_matrix = np.random.uniform(
            low=-1,
            high=1,
            size=(3*_input_size,_input_size),
        )
        hh_matrix = np.random.uniform(
            low=-1,
            high=1,
            size=(3*_input_size,_input_size),
        )
        ih_bias = np.random.uniform(low=-1, high=1, size=(3*_input_size,))
        hh_bias = np.random.uniform(low=-1, high=1, size=(3*_input_size,))
        rnn_weights_numpy.extend([ih_matrix, hh_matrix, ih_bias, hh_bias])
    rnn_weights_numpy = [w.astype(np.float32) for w in rnn_weights_numpy]
    rnn_weights_lbann = [
        lbann.Weights(
            initializer=lbann.ValueInitializer(
                values=np.nditer(w, order='F')))
        for w in rnn_weights_numpy
    ]

    # LBANN implementation
    x = x_lbann
    h = h_lbann
    y = lbann.GRU(
        x,
        h,
        hidden_size=_input_size,
        num_layers=_num_layers,
        weights=rnn_weights_lbann,
    )
    z = lbann.L2Norm2(y)
    obj.append(z)
    metrics.append(lbann.Metric(z, name='Multi-layer, unidirectional'))

    # NumPy implementation
    vals = []
    for i in range(num_samples()):
        input_ = get_sample(i).astype(np.float64)
        x = input_[:_sequence_length*_input_size].reshape((_sequence_length,_input_size))
        h = input_[_sequence_length*_input_size:].reshape((_num_layers,_input_size))
        y = numpy_gru(x, h, rnn_weights_numpy)
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
    # Single-layer, unidirectional GRU
    # ------------------------------------------

    # Weights
    rnn_weights_numpy = []
    hidden_size = 7
    ih_matrix = np.random.uniform(
        low=-1,
        high=1,
        size=(3*hidden_size,_input_size),
    )
    hh_matrix = np.random.uniform(
        low=-1,
        high=1,
        size=(3*hidden_size,hidden_size),
    )
    ih_bias = np.random.uniform(low=-1, high=1, size=(3*hidden_size,))
    hh_bias = np.random.uniform(low=-1, high=1, size=(3*hidden_size,))
    rnn_weights_numpy.extend([ih_matrix, hh_matrix, ih_bias, hh_bias])
    rnn_weights_numpy = [w.astype(np.float32) for w in rnn_weights_numpy]
    rnn_weights_lbann = [
        lbann.Weights(
            initializer=lbann.ValueInitializer(
                values=np.nditer(w, order='F')))
        for w in rnn_weights_numpy
    ]

    # LBANN implementation
    x = x_lbann
    h = h_lbann
    h = lbann.Reshape(
        lbann.Slice(
            lbann.Reshape(h, dims=[-1]),
            slice_points=[0, hidden_size],
        ),
        dims=[1, -1],
    )
    y = lbann.GRU(
        x,
        h,
        hidden_size=hidden_size,
        num_layers=1,
        weights=rnn_weights_lbann,
    )
    z = lbann.L2Norm2(y)
    obj.append(z)
    metrics.append(lbann.Metric(z, name='Single-layer, unidirectional'))

    # NumPy implementation
    vals = []
    for i in range(num_samples()):
        input_ = get_sample(i).astype(np.float64)
        x = input_[:_sequence_length*_input_size].reshape((_sequence_length,_input_size))
        h = input_[_sequence_length*_input_size:].reshape((_num_layers,_input_size))
        h = h.flatten()[:hidden_size].reshape((1,hidden_size))
        y = numpy_gru(x, h, rnn_weights_numpy)
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
for _test_func in tools.create_tests(setup_experiment, __file__):
    globals()[_test_func.__name__] = _test_func
