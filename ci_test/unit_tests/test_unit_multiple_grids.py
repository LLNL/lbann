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
sys.path.insert(0, os.path.join(os.path.dirname(current_dir), "common_python"))
import tools

# ==============================================
# Objects for Python data reader
# ==============================================
# Note: The Python data reader imports this file as a module and calls
# the functions below to ingest data.

# Data
np.random.seed(20220118)
_num_samples = 27
_sample_size = 5
_samples = np.random.normal(size=(_num_samples, _sample_size)).astype(np.float32)


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


def setup_experiment(lbann, weekly):
    """Construct LBANN experiment.

    Args:
        lbann (module): Module for LBANN Python frontend

    """
    pytest.skip("skipping test")
    mini_batch_size = num_samples() // 2
    trainer = lbann.Trainer(mini_batch_size)
    model = construct_model(lbann)
    data_reader = construct_data_reader(lbann)
    optimizer = lbann.NoOptimizer()
    return (
        trainer,
        model,
        data_reader,
        optimizer,
        None,
    )  # Don't request any specific number of nodes


def construct_model(lbann):
    """Construct LBANN model.

    Args:
        lbann (module): Module for LBANN Python frontend

    """

    # Input data
    # Note: Sum with a weights layer so that gradient checking will
    # verify that error signals are correct.
    x_weights = lbann.Weights(
        optimizer=lbann.SGD(),
        initializer=lbann.ConstantInitializer(value=0.0),
        name="input_weights",
    )
    x = lbann.Sum(
        lbann.Reshape(lbann.Input(data_field="samples"), dims=_sample_size),
        lbann.WeightsLayer(weights=x_weights, dims=_sample_size),
    )
    x_lbann = x

    # Objects for LBANN model
    obj = []
    metrics = []
    callbacks = []

    # ------------------------------------------
    # Data-parallel layout
    # ------------------------------------------

    # LBANN implementation
    w = lbann.Weights(
        optimizer=lbann.SGD(),
        initializer=lbann.ConstantInitializer(value=0.0),
        name="grid1_dataparallel_weights",
    )
    w = lbann.WeightsLayer(
        weights=w,
        dims=_sample_size,
        data_layout="data_parallel",
        parallel_strategy={"grid_tag": 1},
    )
    x = lbann.Identity(x_lbann, data_layout="data_parallel")
    y1 = lbann.Sin(x, data_layout="data_parallel", parallel_strategy={"grid_tag": 2})
    y2 = lbann.Square(
        y1, data_layout="data_parallel", parallel_strategy={"grid_tag": 1}
    )
    y2 = lbann.Sum(w, y2, data_layout="data_parallel")
    y3 = lbann.Scale(
        y2, constant=2, data_layout="data_parallel", parallel_strategy={"grid_tag": 2}
    )
    y = lbann.Identity(
        y3, data_layout="data_parallel", parallel_strategy={"grid_tag": 0}
    )
    z = lbann.L2Norm2(y)
    obj.append(z)
    metrics.append(lbann.Metric(z, name="data-parallel layout"))

    # NumPy implementation
    vals = []
    for i in range(num_samples()):
        x = get_sample(i).astype(np.float64)
        y = 2 * np.sin(x) ** 2
        z = tools.numpy_l2norm2(y)
        vals.append(z)
    val = np.mean(vals)
    tol = 8 * val * np.finfo(np.float32).eps
    callbacks.append(
        lbann.CallbackCheckMetric(
            metric=metrics[-1].name,
            lower_bound=val - tol,
            upper_bound=val + tol,
            error_on_failure=True,
            execution_modes="test",
        )
    )

    # ------------------------------------------
    # Model-parallel layout
    # ------------------------------------------

    # LBANN implementation
    w = lbann.Weights(
        optimizer=lbann.SGD(),
        initializer=lbann.ConstantInitializer(value=0.5),
        name="grid2_modelparallel_weights",
    )
    x = lbann.Identity(x_lbann, data_layout="model_parallel")
    y1 = lbann.Cos(x, data_layout="model_parallel", parallel_strategy={"grid_tag": 1})
    y2 = lbann.Scale(
        y1, constant=2, data_layout="model_parallel", parallel_strategy={"grid_tag": 2}
    )
    y2 = lbann.FullyConnected(
        y2, num_neurons=1, weights=w, data_layout="model_parallel"
    )
    y3 = lbann.Square(
        y2, data_layout="data_parallel", parallel_strategy={"grid_tag": 1}
    )
    y = lbann.Identity(
        y3, data_layout="data_parallel", parallel_strategy={"grid_tag": 0}
    )
    z = lbann.L2Norm2(y)
    obj.append(z)
    metrics.append(lbann.Metric(z, name="model-parallel layout"))

    # NumPy implementation
    vals = []
    for i in range(num_samples()):
        x = get_sample(i).astype(np.float64)
        y = np.sum(np.cos(x)) ** 2
        z = tools.numpy_l2norm2(y)
        vals.append(z)
    val = np.mean(vals)
    tol = 8 * val * np.finfo(np.float32).eps
    callbacks.append(
        lbann.CallbackCheckMetric(
            metric=metrics[-1].name,
            lower_bound=val - tol,
            upper_bound=val + tol,
            error_on_failure=True,
            execution_modes="test",
        )
    )

    # ------------------------------------------
    # Gradient checking
    # ------------------------------------------

    callbacks.append(lbann.CallbackCheckGradients(error_on_failure=True))

    # ------------------------------------------
    # Construct model
    # ------------------------------------------

    num_epochs = 0
    return lbann.Model(
        num_epochs,
        layers=lbann.traverse_layer_graph(x_lbann),
        objective_function=obj,
        metrics=metrics,
        callbacks=callbacks,
    )


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
    message.reader.extend(
        [
            tools.create_python_data_reader(
                lbann, current_file, "get_sample", "num_samples", "sample_dims", "train"
            )
        ]
    )
    message.reader.extend(
        [
            tools.create_python_data_reader(
                lbann, current_file, "get_sample", "num_samples", "sample_dims", "test"
            )
        ]
    )
    return message


# ==============================================
# Setup PyTest
# ==============================================

# Create test functions that can interact with PyTest
for _test_func in tools.create_tests(
    setup_experiment, __file__, environment={"LBANN_NUM_SUBGRIDS": 2}
):
    globals()[_test_func.__name__] = _test_func
