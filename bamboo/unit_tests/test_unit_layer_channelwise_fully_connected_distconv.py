import functools
import operator
import os
import os.path
import sys
import numpy as np
import pytest 

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
np.random.seed(20200113)
_num_samples = 17
_sample_dims = (8, 1, 3)
_sample_size = functools.reduce(operator.mul, _sample_dims)
_samples = np.random.normal(size=(_num_samples, _sample_size)).astype(np.float32)
_scale = np.random.normal(loc=1, size=(_sample_dims[0], 1, 1)).astype(np.float32)
_bias = np.random.normal(loc=0, size=(_sample_dims[0], 1, 1)).astype(np.float32)


# Sample access functions
def get_sample(index):
    return _samples[index, :]


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
    mini_batch_size = num_samples() // 2
    trainer = lbann.Trainer(mini_batch_size)
    model = construct_model(lbann)
    data_reader = construct_data_reader(lbann)
    optimizer = lbann.NoOptimizer()
    return trainer, model, data_reader, optimizer


def create_parallel_strategy(num_channel_groups):
    return {"channel_groups": num_channel_groups,
            "filter_groups": num_channel_groups }


def construct_model(lbann):
    """Construct LBANN model.

    Args:
        lbann (module): Module for LBANN Python frontend

    """

    # Input data
    # Note: Sum with a weights layer so that gradient checking will
    # verify that error signals are correct.
    x_weights = lbann.Weights(optimizer=lbann.SGD(),
                              initializer=lbann.ConstantInitializer(value=0.0),
                              name='input_weights')
    x0 = lbann.WeightsLayer(weights=x_weights,
                            dims=tools.str_list(_sample_dims))
    x1 = lbann.Reshape(lbann.Input(), dims=tools.str_list(_sample_dims))
    x = lbann.Sum(x0, x1)
    x_lbann = x

    # Objects for LBANN model
    obj = []
    metrics = []
    callbacks = []

    num_height_groups = tools.gpus_per_node(lbann)
    if num_height_groups == 0:
        e = 'this test requires GPUs.'
        print('Skip - ' + e)
        pytest.skip(e)
    # ------------------------------------------
    # Compute expected metric values with NumPy
    # ------------------------------------------

    # Input and output dimensions
    input_channel_dims = _sample_dims[1:]
    output_channel_dims = (1, 2)
    input_channel_size = functools.reduce(operator.mul, input_channel_dims)
    output_channel_size = functools.reduce(operator.mul, output_channel_dims)

    # Weight values
    linearity = np.random.normal(
        size=(output_channel_size, input_channel_size)
    ).astype(np.float32)
    bias = np.random.normal(size=(output_channel_size, 1)).astype(np.float32)

    # With bias

    x = (_samples
         .reshape((-1, input_channel_size))
         .transpose()
         .astype(np.float64))

    y = np.matmul(linearity.astype(np.float64), x) + bias.astype(np.float64)

    print(linearity)
    print(bias)
    np.save("linearity.npy", linearity)
    np.save("bias.npy", bias)
    z = tools.numpy_l2norm2(y) / _num_samples
    val_with_bias = z

    # Without bias
    x = (_samples
         .reshape((-1, input_channel_size))
         .transpose()
         .astype(np.float64))
    y = np.matmul(linearity.astype(np.float64), x)
    z = tools.numpy_l2norm2(y) / _num_samples
    val_without_bias = z

    # ------------------------------------------
    # Data-parallel distconv layout, non-transpose, bias
    # ------------------------------------------

    # LBANN implementation
    linearity_weights = lbann.Weights(
        optimizer=lbann.SGD(),
        initializer=lbann.ValueInitializer(
            values=tools.str_list(np.nditer(linearity, order='F'))
        )
    )
    bias_weights = lbann.Weights(
        optimizer=lbann.SGD(),
        initializer=lbann.ValueInitializer(
            values=tools.str_list(np.nditer(bias))
        )
    )
    x = x_lbann
    y = lbann.ChannelwiseFullyConnected(
        x,
        weights=(linearity_weights, bias_weights),
        output_channel_dims=output_channel_dims,
        parallel_strategy=create_parallel_strategy(num_height_groups),
        name="bias"
    )
    z = lbann.L2Norm2(y)
    obj.append(z)
    metrics.append(lbann.Metric(z, name='data-parallel layout, non-transpose, bias'))

    # NumPy implementation
    tol = 8 * val_with_bias * np.finfo(np.float32).eps
    callbacks.append(lbann.CallbackCheckMetric(
        metric=metrics[-1].name,
        lower_bound=val_with_bias - tol,
        upper_bound=val_with_bias + tol,
        error_on_failure=True,
        execution_modes='test'))

    # ------------------------------------------
    # Data-parallel distconv layout, non-transpose, no bias
    # ------------------------------------------

    # LBANN implementation
    linearity_weights = lbann.Weights(
        optimizer=lbann.SGD(),
        initializer=lbann.ValueInitializer(
            values=tools.str_list(np.nditer(linearity, order='F'))
        )
    )

    x = x_lbann
    y = lbann.ChannelwiseFullyConnected(
        x,
        weights=(linearity_weights),
        output_channel_dims=output_channel_dims,
        parallel_strategy=create_parallel_strategy(num_height_groups),
        name="no_bias"
    )
    z = lbann.L2Norm2(y)
    obj.append(z)
    metrics.append(lbann.Metric(z, name='data-parallel layout, non-transpose, no bias'))

    # NumPy implementation
    tol = 8 * val_without_bias * np.finfo(np.float32).eps
    callbacks.append(lbann.CallbackCheckMetric(
        metric=metrics[-1].name,
        lower_bound=val_without_bias - tol,
        upper_bound=val_without_bias + tol,
        error_on_failure=True,
        execution_modes='test'))

    # ------------------------------------------
    # Dump Outputs and weights
    # ------------------------------------------

    # callbacks.append(lbann.CallbackDumpOutputs(layers=tools.str_list(["bias", "no_bias"]), directory="outputs"))

    # callbacks.append(lbann.CallbackDumpWeights(directory="outputs"))
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
