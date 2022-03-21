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
np.random.seed(20190815)
_num_samples = 29
_sample_dims = (7,5,3)
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
    # Note: We want to use gradient checking to verify that error
    # signals are correct. To do this, we zero-initialize a weights
    # object, construct a zero-valued tensor, and add it to the
    # input. To make sure that batchnorm is non-trivial, we multiply
    # the zero-valued tensor by the mini-batch index.
    x = lbann.Reshape(lbann.Input(data_field='samples'), dims=tools.str_list(_sample_dims))
    x_weights = lbann.Weights(optimizer=lbann.SGD(),
                              initializer=lbann.ConstantInitializer(value=0.0),
                              name='input_weights')
    x0 = lbann.WeightsLayer(weights=x_weights,
                            dims=tools.str_list(_sample_dims))
    x1 = lbann.Divide(lbann.MiniBatchIndex(), lbann.MiniBatchSize())
    x1 = lbann.Tessellate(lbann.Reshape(x1, dims='1 1 1'), dims=tools.str_list(_sample_dims))
    x = lbann.Sum(x, lbann.Multiply(x0, x1))
    x_lbann = x

    # Objects for LBANN model
    obj = []
    metrics = []
    callbacks = []

    # ------------------------------------------
    # Data-parallel layout
    # ------------------------------------------

    # LBANN implementation
    decay = 0.9
    epsilon = 1e-5
    x = x_lbann
    y = lbann.EntrywiseBatchNormalization(x,
                                          decay=decay,
                                          epsilon=epsilon,
                                          data_layout='data_parallel')
    z = lbann.L2Norm2(y)
    obj.append(z)
    metrics.append(lbann.Metric(z, name='data-parallel layout'))

    # ------------------------------------------
    # Model-parallel layout
    # ------------------------------------------

    # LBANN implementation
    decay = 0.9
    epsilon = 1e-5
    x = x_lbann
    y = lbann.EntrywiseBatchNormalization(x,
                                          decay=decay,
                                          epsilon=epsilon,
                                          data_layout='model_parallel')
    z = lbann.L2Norm2(y)
    obj.append(z)
    metrics.append(lbann.Metric(z, name='model-parallel layout'))

    # ------------------------------------------
    # Gradient checking
    # ------------------------------------------

    callbacks.append(lbann.CallbackCheckGradients(error_on_failure=True))

    # ------------------------------------------
    # Construct model
    # ------------------------------------------

    num_epochs = 1
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
