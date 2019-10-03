import functools
import operator
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
one_hot_size = 7
seed = 201909113

# Sample access functions
def get_sample(index):
    np.random.seed(seed+index)
    return [np.random.uniform(-1, one_hot_size+1)]
def num_samples():
    return 47
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
    y1 = lbann.OneHot(x, size=one_hot_size)
    y2 = lbann.Concatenation([lbann.Constant(value=i+1, num_neurons='1')
                              for i in range(one_hot_size)])
    y = lbann.Multiply([y1, y2])
    z = lbann.L2Norm2(y)

    # Objects for LBANN model
    layers = list(lbann.traverse_layer_graph(x))
    metric = lbann.Metric(z, name='obj')
    obj = lbann.ObjectiveFunction(z)
    callbacks = []

    # Compute expected metric value
    vals = []
    for i in range(num_samples()):
        x = get_sample(i)[0]
        y = int(x) + 1 if (0 <= x and x < one_hot_size) else 0
        z = y * y
        vals.append(z)
    val = np.mean(vals)
    tol = 8 * val * np.finfo(np.float32).eps
    callbacks.append(lbann.CallbackCheckMetric(
        metric=metric.name,
        lower_bound=val-tol,
        upper_bound=val+tol,
        error_on_failure=True,
        execution_modes='test'))

    # Construct model
    mini_batch_size = 19
    num_epochs = 0
    return lbann.Model(mini_batch_size,
                       num_epochs,
                       layers=layers,
                       objective_function=obj,
                       metrics=[metric],
                       callbacks=callbacks)

def construct_data_reader(lbann):
    """Construct Protobuf message for Python data reader.

    The Python data reader will import the current Python file to
    access the sample access functions.

    Args:
        lbann (module): Module for LBANN Python frontend

    """
    module_name = os.path.splitext(os.path.basename(current_file))[0]

    # Base data reader message
    message = lbann.reader_pb2.DataReader()

    # Training set data reader
    # TODO: This can be removed once
    # https://github.com/LLNL/lbann/issues/1098 is resolved.
    data_reader = message.reader.add()
    data_reader.name = 'python'
    data_reader.role = 'train'
    data_reader.percent_of_data_to_use = 1.0
    data_reader.python.module = module_name
    data_reader.python.module_dir = current_dir
    data_reader.python.sample_function = 'get_sample'
    data_reader.python.num_samples_function = 'num_samples'
    data_reader.python.sample_dims_function = 'sample_dims'

    # Test set data reader
    data_reader = message.reader.add()
    data_reader.name = 'python'
    data_reader.role = 'test'
    data_reader.percent_of_data_to_use = 1.0
    data_reader.python.module = module_name
    data_reader.python.module_dir = current_dir
    data_reader.python.sample_function = 'get_sample'
    data_reader.python.num_samples_function = 'num_samples'
    data_reader.python.sample_dims_function = 'sample_dims'

    return message

# ==============================================
# Setup PyTest
# ==============================================

# Create test functions that can interact with PyTest
# Note: Create test name by removing ".py" from file name
_test_name = os.path.splitext(os.path.basename(current_file))[0]
for test in tools.create_tests(setup_experiment, _test_name):
    globals()[test.__name__] = test
