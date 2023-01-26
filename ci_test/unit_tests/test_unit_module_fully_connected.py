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
np.random.seed(20210917)
_num_samples = 3
_input_size = 5
_samples = np.random.normal(size=(_num_samples,2,_input_size)).astype(np.float32)

# Sample access functions
def get_sample(index):
    return _samples[index].flatten()
def num_samples():
    return _num_samples
def sample_dims():
    return (2*_input_size,)

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

    # ------------------------------------------
    # NumPy implementation
    # ------------------------------------------

    w_np = np.random.normal(size=(1,_input_size)).astype(np.float32)
    b_np = np.random.normal(size=(1,1)).astype(np.float32)
    w = w_np.astype(np.float64)
    b = b_np.astype(np.float64)
    x0 = _samples[:,0,:].astype(np.float64)
    x1 = _samples[:,1,:].astype(np.float64)
    y0 = np.tanh(np.matmul(x0, w.transpose()) + b)
    y1 = np.tanh(np.matmul(x1, w.transpose()) + b)
    y0_np = np.mean(y0)
    y1_np = np.mean(y1)

    # ------------------------------------------
    # LBANN implementation
    # ------------------------------------------

    # Objects for LBANN model
    metrics = []
    callbacks = []

    # Input data
    x = lbann.Slice(
        lbann.Input(data_field='samples'),
        slice_points=[0, _input_size, _input_size*2],
    )
    x0 = lbann.Identity(x)
    x1 = lbann.Identity(x)

    # Fully-connected module
    import lbann.modules
    fc = lbann.modules.FullyConnectedModule(
        1,
        bias=True,
        weights=[
            lbann.Weights(
                initializer=lbann.ValueInitializer(
                    values=np.nditer(w_np))),
            lbann.Weights(
                initializer=lbann.ValueInitializer(
                    values=np.nditer(b_np))),
        ],
        activation=lbann.Tanh,
    )

    # y1
    y1 = fc(x1)
    tol = abs(8 * y0_np * np.finfo(np.float32).eps)
    metrics.append(lbann.Metric(y1, name='y1'))
    callbacks.append(lbann.CallbackCheckMetric(
        metric=metrics[-1].name,
        lower_bound=y1_np-tol,
        upper_bound=y1_np+tol,
        error_on_failure=True,
        execution_modes='test'))

    # y0
    y0 = fc(x0)
    tol = abs(8 * y0_np * np.finfo(np.float32).eps)
    metrics.append(lbann.Metric(y0, name='y0'))
    callbacks.append(lbann.CallbackCheckMetric(
        metric=metrics[-1].name,
        lower_bound=y0_np-tol,
        upper_bound=y0_np+tol,
        error_on_failure=True,
        execution_modes='test'))

    # ------------------------------------------
    # Construct model
    # ------------------------------------------

    num_epochs = 0
    return lbann.Model(num_epochs,
                       layers=lbann.traverse_layer_graph(x),
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
