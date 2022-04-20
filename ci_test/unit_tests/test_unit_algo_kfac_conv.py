import functools
import math
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
np.random.seed(20211119)
_num_samples = 5
_in_size = 3
_out_size = 2
_samples = np.random.normal(size=(_num_samples,_in_size+_out_size)).astype(np.float32)

# Sample access functions
def get_sample(index):
    return _samples[index,:]
def num_samples():
    return _num_samples
def sample_dims():
    return (_in_size+_out_size,)

# ==============================================
# Setup LBANN experiment
# ==============================================

def setup_experiment(lbann, weekly):
    """Construct LBANN experiment.

    Args:
        lbann (module): Module for LBANN Python frontend

    """
    trainer = construct_trainer(lbann)
    model = construct_model(lbann)
    data_reader = construct_data_reader(lbann)
    optimizer = lbann.NoOptimizer()
    return trainer, model, data_reader, optimizer, None # Don't request any specific number of nodes

def construct_trainer(lbann, ):
    """Construct LBANN trainer and training algorithm.

    Args:
        lbann (module): Module for LBANN Python frontend

    """
    num_epochs = 1
    mini_batch_size = num_samples()
    algo = lbann.KFAC(
        "kfac",
        lbann.BatchedIterativeOptimizer("sgd", epoch_count=num_epochs),
        damping_warmup_steps=0,
        kronecker_decay=0,
        damping_act="1e-30",
        damping_err="1e-30",
        compute_interval=1,
    )
    trainer = lbann.Trainer(mini_batch_size, training_algo=algo)
    return trainer

def construct_model(lbann):
    """Construct LBANN model.

    Args:
        lbann (module): Module for LBANN Python frontend

    """

    # ------------------------------------------
    # NumPy implementation
    # ------------------------------------------

    # Compute Kronecker factors
    x = _samples[:,:_in_size]
    dy = _samples[:,-_out_size:] / _num_samples
    A = np.matmul(x.T, x)
    G = np.matmul(dy.T, dy)

    # Apply one K-FAC step
    w = np.ones((_out_size, _in_size))
    dw = np.matmul(dy.T, x)
    dw = np.linalg.solve(A.T, dw.T).T
    dw = np.linalg.solve(G, dw)
    w_next = w - dw

    # ------------------------------------------
    # Basic model with conv layer
    # ------------------------------------------

    # Inputs and error signals are from data reader
    input_ = lbann.Input(data_field='samples')
    input_slice = lbann.Slice(input_, slice_points=[0,_in_size,_in_size+_out_size])
    x = lbann.Reshape(input_slice, dims=[-1,1,1])
    dy = lbann.Reshape(input_slice, dims=[-1,1,1])

    # Convolution layer
    w = lbann.Weights(
        initializer=lbann.ConstantInitializer(value=1),
        optimizer=lbann.SGD(learn_rate=1),
    )
    y = lbann.Convolution(
        x,
        weights=w,
        num_dims=2,
        num_output_channels=_out_size,
        num_groups=1,
        has_vectors=False,
        conv_dims_i=1,
        conv_pads_i=0,
        conv_strides_i=1,
        conv_dilations_i=1,
        has_bias=False,
    )
    obj = lbann.Reduction(lbann.Multiply(y, dy))

    # ------------------------------------------
    # Metric checking
    # ------------------------------------------

    # Extract weights entries
    w = lbann.WeightsLayer(
        weights=w,
        dims=[_out_size,_in_size,1,1],
    )
    w = lbann.Slice(
        lbann.Reshape(w, dims=[-1]),
        slice_points=range(_out_size*_in_size+1),
    )
    w = [lbann.Identity(w) for _ in range(_out_size*_in_size)]

    # Metric checking with values from NumPy implementation
    metrics= []
    callbacks = []
    tol = 8 * np.finfo(np.float32).eps
    for i in range(_out_size*_in_size):
        val = w_next.flatten()[i]
        metrics.append(lbann.Metric(w[i], name=f'w[{i}]'))
        callbacks.append(lbann.CallbackCheckMetric(
            metric=metrics[-1].name,
            lower_bound=val-tol,
            upper_bound=val+tol,
            error_on_failure=True,
            execution_modes='test'))

    # ------------------------------------------
    # Construct model
    # ------------------------------------------

    num_epochs = 1
    return lbann.Model(
        num_epochs,
        layers=lbann.traverse_layer_graph([input_] + w),
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
for _test_func in tools.create_tests(
        setup_experiment,
        __file__,
        environment={"LBANN_KEEP_ERROR_SIGNALS": 1},
):
    globals()[_test_func.__name__] = _test_func
