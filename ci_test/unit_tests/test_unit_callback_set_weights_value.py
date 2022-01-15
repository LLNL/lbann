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
np.random.seed(20200526)
_samples = np.random.uniform(size=13).astype(np.float32)

# Sample access functions
def get_sample(index):
    return (_samples[index],)
def num_samples():
    return len(_samples)
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
    trainer = lbann.Trainer(mini_batch_size=1)
    model = construct_model(lbann)
    data_reader = construct_data_reader(lbann)
    optimizer = lbann.NoOptimizer()
    return trainer, model, data_reader, optimizer

def construct_model(lbann):
    """Construct LBANN model.

    Args:
        lbann (module): Module for LBANN Python frontend

    """

    # Objects for LBANN model
    obj = []
    metrics = []
    callbacks = []

    # LBANN implementation
    weights_values = np.random.uniform(size=num_samples()).astype(np.float32)
    w = lbann.Weights(optimizer=None,
                      initializer=lbann.ConstantInitializer(value=1234.5))
    for step, val in enumerate(weights_values):
        callbacks.append(
            lbann.CallbackSetWeightsValue(weights=w.name, value=val, step=step)
        )
    x_lbann = lbann.Input(data_field='samples')
    x = x_lbann
    y = lbann.WeightsLayer(weights=w, dims='1')
    z = lbann.Multiply(x, y)
    metrics.append(lbann.Metric(z, name='value'))

    # Numpy implementation of training
    vals = []
    for step, val in enumerate(weights_values):
        x = np.float64(get_sample(step)[0])
        y = np.float64(val)
        z = x * y
        vals.append(z)
    val = np.mean(vals)
    tol = 8 * val * np.finfo(np.float32).eps
    callbacks.append(lbann.CallbackCheckMetric(
        metric=metrics[-1].name,
        lower_bound=val-tol,
        upper_bound=val+tol,
        error_on_failure=True,
        execution_modes='train'))

    # Numpy implementation of testing
    vals = []
    for i in range(num_samples()):
        x = np.float64(get_sample(i)[0])
        y = np.float64(weights_values[-1])
        z = x * y
        vals.append(z)
    val = np.mean(vals)
    tol = 8 * val * np.finfo(np.float32).eps
    callbacks.append(lbann.CallbackCheckMetric(
        metric=metrics[-1].name,
        lower_bound=val-tol,
        upper_bound=val+tol,
        error_on_failure=True,
        execution_modes='test'))

    # Construct model
    return lbann.Model(epochs=1,
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
### @todo Run on >1 proc when https://github.com/LLNL/lbann/issues/1548 is resolved
for _test_func in tools.create_tests(setup_experiment, __file__, procs_per_node=1, nodes=1):
    globals()[_test_func.__name__] = _test_func
