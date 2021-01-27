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
input_size = 11
output_size = 15
seed = 20210127

# Sample access functions
def get_sample(index):
    np.random.seed(seed+index)
    values = [np.random.normal() for _ in range(input_size)]
    indices = [
        np.random.uniform(-1, output_size+1)
        for _ in range(input_size)
    ]
    return values + indices
def num_samples():
    return 23
def sample_dims():
    return (2*input_size,)

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
    optimizer = lbann.NoOptimizer()
    return trainer, model, data_reader, optimizer

def construct_model(lbann):
    """Construct LBANN model.

    Args:
        lbann (module): Module for LBANN Python frontend

    """

    # Input data
    # Note: Sum with a weights layer so that gradient checking will
    # verify that error signals are correct.
    x = lbann.Identity(lbann.Input())
    x_slice = lbann.Slice(
        x,
        slice_points=tools.str_list([0,input_size,2*input_size]),
    )
    x0_weights = lbann.Weights(
        optimizer=lbann.SGD(),
        initializer=lbann.ConstantInitializer(value=0.0),
        name='input_weights',
    )
    x0 = lbann.Sum(
        lbann.Identity(x_slice),
        lbann.WeightsLayer(weights=x0_weights, dims=tools.str_list(input_size)),
    )
    x1 = lbann.Identity(x_slice)

    # Apply scatter
    y0 = lbann.Scatter(x0, x1, dims=tools.str_list(output_size))
    y0.device = 'CPU' ### @todo Remove
    y1 = lbann.Concatenation([
        lbann.Constant(value=i+1, num_neurons='1')
        for i in range(output_size)
    ])
    y = lbann.Multiply(y0, y1)
    z = lbann.L2Norm2(y)

    # Objects for LBANN model
    layers = list(lbann.traverse_layer_graph(x))
    metric = lbann.Metric(z, name='obj')
    obj = lbann.ObjectiveFunction(z)
    callbacks = []

    # Compute expected metric value
    vals = []
    for i in range(num_samples()):
        x = get_sample(i)
        x0 = x[:input_size]
        x1 = x[input_size:]
        y0 = np.zeros(output_size)
        for i in range(input_size):
            if 0 <= x1[i] < output_size:
                y0[int(x1[i])] += x0[i]
        z = 0
        for i in range(output_size):
            z += ((i+1)*y0[i]) ** 2
        vals.append(z)
    val = np.mean(vals)
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
for test in tools.create_tests(setup_experiment, __file__):
    globals()[test.__name__] = test
