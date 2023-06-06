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
_num_samples = 16
_num_channels = 9
_input_size = 5
_hidden_size = 7
_sample_size = _num_channels*_input_size + _num_channels *_hidden_size
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

def numpy_gru_cell(x, h, w):
    #
    # This implements a 2 dimensional analogue of the PyTorch.nn.GRUCell
    # See here for more details:
    # https://pytorch.org/docs/stable/generated/torch.nn.GRUCell.html#torch.nn.GRUCell
    #
    # Dimensions
    input_size = x[0].size
    hidden_size = h[0].size

    # Unroll GRU

    for sample in range(x.shape[0]):
        ih = np.matmul(w[0], x[sample]) + w[1]
        hh = np.matmul(w[2], h[sample]) + w[3]
        r = scipy.special.expit(ih[:hidden_size] + hh[:hidden_size])
        z = scipy.special.expit(ih[hidden_size:2*hidden_size] + hh[hidden_size:2*hidden_size])
        n = np.tanh(ih[2*hidden_size:] + r*hh[2*hidden_size:])
        h[sample] = (1-z)*n + z*h[sample]
    return h

# ==============================================
# Setup LBANN experiment
# ==============================================

def setup_experiment(lbann, weekly):
    """Construct LBANN experiment.

    Args:
        lbann (module): Module for LBANN Python frontend

    """

    # Skip test on non-GPU systems
    # Note: Test requires cuDNN (on GPU) or oneDNN (on CPU).
    ### @todo Assume LBANN has been built with oneDNN?
    if not tools.gpus_per_node(lbann):
        message = f'{os.path.basename(__file__)} requires cuDNN or oneDNN'
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
    from lbann.modules.rnn import ChannelwiseGRU

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
        slice_points=[0, _num_channels*_input_size, _sample_size],
    )
    x = lbann.Reshape(input_slice, dims=[_num_channels,_input_size], name="input_reshape")
    x = lbann.Sum(x, lbann.WeightsLayer(weights=x_weights, dims=[_num_channels,_input_size]), name="input_sum")

    h = lbann.Reshape(input_slice, dims=[_num_channels,_hidden_size],name="hidden_reshape")
    h = lbann.Sum(h, lbann.WeightsLayer(weights=h_weights, dims=[_num_channels,_hidden_size]), name="input_hidden_sum")

    x_lbann = x
    h_lbann = h


    # Objects for LBANN model
    obj = []
    metrics = []
    callbacks = []

    # Weights
    rnn_weights_numpy = []
    ih_matrix = np.random.uniform(
        low=-1,
        high=1,
        size=(3*_hidden_size,_input_size),
    )
    hh_matrix = np.random.uniform(
        low=-1,
        high=1,
        size=(3*_hidden_size,_hidden_size),
    )
    ih_bias = np.random.uniform(low=-1, high=1, size=(3*_hidden_size,))
    hh_bias = np.random.uniform(low=-1, high=1, size=(3*_hidden_size,))
    rnn_weights_numpy.extend([ih_matrix, ih_bias, hh_matrix, hh_bias])

    rnn_weights_lbann = [
        lbann.Weights(
            initializer=lbann.ValueInitializer(
                values=np.nditer(w, order='F')))
        for w in rnn_weights_numpy
    ]

    # LBANN implementation
    x = x_lbann
    h = h_lbann
    channelwise_GRU_cell = ChannelwiseGRU(num_channels=_num_channels,
                                          size=_hidden_size,
                                          weights=rnn_weights_lbann)
    y = channelwise_GRU_cell(x, h)
    z = lbann.L2Norm2(y)
    obj.append(z)
    metrics.append(lbann.Metric(z, name="Multi-channel, Unidirectional, GRU Cell"))

    # NumPy implementation
    vals = []
    for i in range(num_samples()):
        input_ = get_sample(i).astype(np.float64)
        x = input_[:_num_channels*_input_size].reshape((_num_channels,_input_size))
        h = input_[_num_channels*_input_size:].reshape((_num_channels,_hidden_size))
        y = numpy_gru_cell(x, h, rnn_weights_numpy)
        z = tools.numpy_l2norm2(y)
        vals.append(z)
    val = np.mean(vals)
    tol = 8 * val * np.finfo(np.float32).eps
    callbacks.append(lbann.CallbackPrintModelDescription())
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
