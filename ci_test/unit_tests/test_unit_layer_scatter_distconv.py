import os
import os.path
import sys
from unittest.mock import call
import numpy as np
import pytest
import lbann.contrib.args

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
width = 2
height = 16
input_size = width * height
output_size = 8
seed = 20220911


def get_sample(index):
    # Sample access functions
    np.random.seed(seed + index)
    values = [np.random.normal() for _ in range(input_size)]
    indices = [np.random.uniform(-1, output_size) for _ in range(height)]
    return values + indices

def num_samples():
    return 5

def sample_dims():
    return (input_size + height,)

def create_parallel_strategy(num_channel_groups):
    return {"channel_groups": num_channel_groups,
            "filter_groups": num_channel_groups}

# ==============================================
# Setup LBANN experiment
# ==============================================


def setup_experiment(lbann, weekly):
    """Construct LBANN experiment.

    Args:
        lbann (module): Module for LBANN Python frontend

    """
    if not (lbann.has_feature('DISTCONV') and lbann.has_feature('NVSHMEM')) :
        message = f'{os.path.basename(__file__)} requires DISTCONV and NVSHMEM'
        print('Skip - ' + message)
        pytest.skip(message)

    mini_batch_size = num_samples() // 2
    trainer = lbann.Trainer(mini_batch_size)
    model = construct_model(lbann)
    data_reader = construct_data_reader(lbann)
    optimizer = lbann.NoOptimizer()
    return trainer, model, data_reader, optimizer, None  # Don't request any specific number of nodes


def construct_model(lbann):
    """Construct LBANN model.

    Args:
        lbann (module): Module for LBANN Python frontend

    """
    # Objects for LBANN model
    obj = []
    metrics = []
    callbacks = []

    # Input data
    # Note: Sum with a weights layer so that gradient checking will
    # verify that error signals are correct.
    x = lbann.Input(data_field='samples')
    x_slice = lbann.Slice(
        x,
        slice_points=[0, input_size, input_size + height],
    )
    x0_weights = lbann.Weights(
        optimizer=lbann.SGD(),
        initializer=lbann.ConstantInitializer(value=0.0),
        name='input_weights',
    )
    x0 = lbann.Sum(
        lbann.Identity(x_slice),
        lbann.WeightsLayer(weights=x0_weights, dims=input_size),
    )

    x1 = lbann.Identity(x_slice)

    ######################################################################
    #
    #          3D Values , 3D Input, Axis = 0, Distconv
    #
    ######################################################################
    num_channel_groups = tools.gpus_per_node(lbann)

    x0 = lbann.Reshape(x0, dims=[height, width, 1], name="values_distconv_axis_0")
    x1 = lbann.Reshape(x1, dims=[height, 1, 1], name="indices_distconv_axis_0")

    # output should be (output_size, width, 1)
    x1 = lbann.Identity(x1, parallel_strategy=create_parallel_strategy(num_channel_groups))

    y0 = lbann.Scatter(x0, x1,
                       dims=[output_size, width, 1],
                       axis=0, name="Scatter_distconv_axis_0",
                       parallel_strategy=create_parallel_strategy(num_channel_groups)) 

    y1 = lbann.Concatenation([
        lbann.Constant(value=i + 1, num_neurons=[1])
        for i in range(output_size * width)])

    y1 = lbann.Reshape(y1, dims=[width * output_size])
    y0 = lbann.Reshape(y0, dims=[width * output_size])

    y = lbann.Multiply(y0, y1)

    z = lbann.L2Norm2(y)

    obj.append(z)
    metrics.append(lbann.Metric(z, name='3D, axis=0'))

    vals = []
    for i in range(num_samples()):
        _x = get_sample(i)
        x0 = np.array(_x[:input_size]).reshape((height, width))
        x1 = np.array(_x[input_size:input_size + height])
        y0 = np.zeros((output_size, width))

        for i in range(height):
            ind = x1[i]
            if  0 <= ind < output_size:
                for j in range(width):
                    y0[int(x1[i])][j] += x0[i][j]
        z = 0
        for i in range(width * output_size):
            z += ((i + 1) * y0.flatten()[i])**2
        vals.append(z)
    val = np.mean(vals)
    tol = 8 * val * np.finfo(np.float32).eps
    callbacks.append(lbann.CallbackCheckMetric(
        metric=metrics[-1].name,
        lower_bound=val - tol,
        upper_bound=val + tol,
        error_on_failure=True,
        execution_modes='test'))
    
    # Gradient checking
    callbacks.append(lbann.CallbackCheckGradients(error_on_failure=True))

    # Construct model
    num_epochs = 0
    layers = list(lbann.traverse_layer_graph(x))

    return lbann.Model(num_epochs,
                       layers=layers,
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
for _test_func in tools.create_tests(setup_experiment, __file__,
                                    environment=lbann.contrib.args.get_distconv_environment(init_nvshmem=True)):
    globals()[_test_func.__name__] = _test_func
