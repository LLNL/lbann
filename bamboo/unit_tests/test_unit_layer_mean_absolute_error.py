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
# Note: MAE is not differentiable when the two inputs match, so we
# make sure inputs have separated values.
np.random.seed(201910248)
_samples = np.random.uniform(-0.25, 0.25, size=(27,2,7)).astype(np.float32)
_samples[:,1,:] += np.random.choice([-1.0,1.0], size=(27,7))

# Sample access functions
def get_sample(index):
    return _samples[index].reshape(-1)
def num_samples():
    return _samples.shape[0]
def sample_dims():
    return (2*_samples.shape[-1],)

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
    # Note: Sum with weights layers so that gradient checking will
    # verify that error signals are correct.
    slice_size = _samples.shape[-1]
    x0_weights = lbann.Weights(optimizer=lbann.SGD(),
                               initializer=lbann.ConstantInitializer(value=0.0),
                               name='input0_weights')
    x1_weights = lbann.Weights(optimizer=lbann.SGD(),
                               initializer=lbann.ConstantInitializer(value=0.0),
                               name='input1_weights')
    x_slice = lbann.Slice(lbann.Input(data_field='datum'),
                          slice_points=tools.str_list([0, slice_size, 2*slice_size]))
    x0 = lbann.Sum(x_slice,
                   lbann.WeightsLayer(weights=x0_weights, dims=str(slice_size)))
    x1 = lbann.Sum(x_slice,
                   lbann.WeightsLayer(weights=x1_weights, dims=str(slice_size)))
    x0_lbann = x0
    x1_lbann = x1

    # Objects for LBANN model
    obj = []
    metrics = []
    callbacks = []

    # ------------------------------------------
    # Data-parallel layout
    # ------------------------------------------

    # LBANN implementation
    x0 = x0_lbann
    x1 = x1_lbann
    y = lbann.MeanAbsoluteError(x0, x1, data_layout='data_parallel')
    z = lbann.L2Norm2(y)
    obj.append(z)
    metrics.append(lbann.Metric(z, name='data-parallel layout'))

    # NumPy implementation
    vals = []
    for i in range(num_samples()):
        x = get_sample(i).astype(np.float64)
        x0 = x[:slice_size]
        x1 = x[slice_size:]
        y = np.linalg.norm(x1-x0, 1) / slice_size
        z = tools.numpy_l2norm2(y)
        vals.append(z)
    val = np.mean(vals)
    tol = 8 * val * np.finfo(np.float32).eps
    callbacks.append(lbann.CallbackCheckMetric(
        metric=metrics[-1].name,
        lower_bound=val-tol,
        upper_bound=val+tol,
        error_on_failure=True,
        execution_modes='test'))

    # ------------------------------------------
    # Model-parallel layout
    # ------------------------------------------

    # LBANN implementation
    x0 = x0_lbann
    x1 = x1_lbann
    y = lbann.MeanAbsoluteError(x0, x1, data_layout='model_parallel')
    z = lbann.L2Norm2(y)
    obj.append(z)
    metrics.append(lbann.Metric(z, name='model-parallel layout'))

    # NumPy implementation
    vals = []
    for i in range(num_samples()):
        x = get_sample(i).astype(np.float64)
        x0 = x[:slice_size]
        x1 = x[slice_size:]
        y = np.linalg.norm(x1-x0, 1) / slice_size
        z = tools.numpy_l2norm2(y)
        vals.append(z)
    val = np.mean(vals)
    tol = 8 * val * np.finfo(np.float32).eps
    callbacks.append(lbann.CallbackCheckMetric(
        metric=metrics[-1].name,
        lower_bound=val-tol,
        upper_bound=val+tol,
        error_on_failure=True,
        execution_modes='test'))

    # ------------------------------------------
    # Gradient checking
    # ------------------------------------------

    callbacks.append(lbann.CallbackCheckGradients(error_on_failure=True))

    # ------------------------------------------
    # Construct model
    # ------------------------------------------

    num_epochs = 0
    return lbann.Model(num_epochs,
                       layers=lbann.traverse_layer_graph(x0_lbann),
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
