import os
import os.path
import sys
import numpy as np

# Local files
current_file = os.path.realpath(__file__)
current_dir = os.path.dirname(current_file)
sys.path.insert(0, os.path.join(os.path.dirname(current_dir), 'common_python'))
import tools

# ==============================================
# Objects for Python data reader
# ==============================================
# Note: The Python data reader imports this file and calls the
# functions below to ingest data. This is the only part of the script
# that should be executed when the script is imported, or else the
# Python data reader might misbehave.

# Data
np.random.seed(20190708)
_num_samples = 23
_sample_size = 7
_samples = np.random.normal(size=(_num_samples,_sample_size))
_samples = _samples.astype(np.float32)

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
    trainer = lbann.Trainer()
    model = construct_model(lbann)
    data_reader = construct_data_reader(lbann)
    optimizer = lbann.NoOptimizer()
    return trainer, model, data_reader, optimizer

def construct_model(lbann):
    """Construct LBANN model.

    Args:
        lbann (module): Module for LBANN Python frontend

    """

    # Layer graph
    x = lbann.Input()
    obj = lbann.L2Norm2(x)
    layers = list(lbann.traverse_layer_graph(x))
    metric = lbann.Metric(obj, name='obj')
    callbacks = []

    # Compute expected value with NumPy
    vals = []
    for i in range(num_samples()):
        x = get_sample(i)
        obj = np.inner(x, x)
        vals.append(obj)
    val = np.mean(vals)
    tol = 8 * val * np.finfo(np.float32).eps
    callbacks.append(lbann.CallbackCheckMetric(
        metric=metric.name,
        lower_bound=val-tol,
        upper_bound=val+tol,
        error_on_failure=True,
        execution_modes='test'))

    # Construct model
    mini_batch_size = 5
    num_epochs = 0
    return lbann.Model(mini_batch_size,
                       num_epochs,
                       layers=layers,
                       metrics=[metric],
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
for test in tools.create_tests(setup_experiment, _test_name):
    globals()[test.__name__] = test
