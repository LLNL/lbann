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
np.random.seed(20210301)
_num_samples = 11
_sample_size = 3
_samples = np.random.normal(size=(_num_samples,_sample_size))
_samples = _samples.astype(np.float32)

# Sample access functions
def get_sample(index):
    return _samples[index]
def num_samples():
    return _samples.shape[0]
def sample_dims():
    return (_samples.shape[1],)

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
    # Note: Multiply with a weights layer so that gradient checking
    # will verify that error signals are correct. We multiply instead
    # of adding so that each batch sample contributes a different
    # gradient.
    x_weights = lbann.Weights(
        optimizer=lbann.SGD(),
        initializer=lbann.ConstantInitializer(value=1.0),
        name='input_weights'
    )
    x = lbann.Multiply(
        lbann.Input(data_field='samples'),
        lbann.WeightsLayer(weights=x_weights, dims=tools.str_list(_sample_size)),
    )

    # Compute variance along batch dimension
    sum_x = lbann.BatchwiseReduceSum(x)
    sum_x2 = lbann.BatchwiseReduceSum(lbann.Square(x))
    mini_batch_size = lbann.Tessellate(lbann.MiniBatchSize(), hint_layer=x)
    mean_x = lbann.Divide(sum_x, mini_batch_size)
    mean_x2 = lbann.Divide(sum_x2, mini_batch_size)
    var = lbann.Subtract(mean_x2, lbann.Square(mean_x))
    obj = lbann.L2Norm2(var)

    # Objects for LBANN model
    layers = list(lbann.traverse_layer_graph(x))
    metric = lbann.Metric(obj, name='obj')
    obj = lbann.ObjectiveFunction(obj)
    callbacks = []

    # Compute expected metric value
    var = np.var(_samples, axis=0)
    val = tools.numpy_l2norm2(var)
    tol = 8 * val * np.finfo(np.float32).eps
    callbacks.append(lbann.CallbackCheckMetric(
        metric=metric.name,
        lower_bound=val-tol,
        upper_bound=val+tol,
        error_on_failure=True,
        execution_modes='test'))

    # Gradient checking
    callbacks.append(lbann.CallbackCheckGradients(error_on_failure=True))

    # Construct model
    num_epochs = 0
    return lbann.Model(num_epochs,
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
