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
np.random.seed(20191111)
_m = 2
_n = 3
_k = 4
_N = 5
_samples = np.random.normal(size=(32,_N*(_m*_k)+_N*(_k*_n))).astype(np.float32)

# Sample access functions
def get_sample(index):
    return _samples[index].reshape(-1)
def num_samples():
    return _samples.shape[0]
def sample_dims():
    return (_samples.shape[-1],)

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
    x0_weights = lbann.Weights(optimizer=lbann.SGD(),
                               initializer=lbann.ConstantInitializer(value=0.0),
                               name='input0_weights')
    x1_weights = lbann.Weights(optimizer=lbann.SGD(),
                               initializer=lbann.ConstantInitializer(value=0.0),
                               name='input1_weights')
    x_slice = lbann.Slice(lbann.Input(),
                          slice_points=tools.str_list([0, _N*_m*_k, _N*_m*_k+_N*_k*_n]))
    x0 = lbann.Sum(x_slice,
                   lbann.WeightsLayer(weights=x0_weights, dims=str(_N*_m*_k)))
    x1 = lbann.Sum(x_slice,
                   lbann.WeightsLayer(weights=x1_weights, dims=str(_N*_k*_n)))
    x0_lbann = x0
    x1_lbann = x1

    # Objects for LBANN model
    obj = []
    metrics = []
    callbacks = []

    # ------------------------------------------
    # NN GEMM
    # ------------------------------------------

    # LBANN implementation
    x0 = lbann.Reshape(x0_lbann, dims=tools.str_list([_N, _m, _k]))
    x1 = lbann.Reshape(x1_lbann, dims=tools.str_list([_N, _k, _n]))
    y = lbann.MatMul(x0, x1, data_layout='data_parallel')
    z = lbann.L2Norm2(y)
    obj.append(z)
    metrics.append(lbann.Metric(z, name='NN GEMM'))

    # NumPy implementation
    vals = []
    for i in range(num_samples()):
        x = get_sample(i).astype(np.float64)
        x0 = x[:_N*_m*_k].reshape([_N,_m,_k])
        x1 = x[_N*_m*_k:].reshape([_N,_k,_n])
        y = np.matmul(x0, x1)
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
    # TN GEMM
    # ------------------------------------------

    # LBANN implementation
    x0 = lbann.Reshape(x0_lbann, dims=tools.str_list([_N, _k, _m]))
    x1 = lbann.Reshape(x1_lbann, dims=tools.str_list([_N, _k, _n]))
    y = lbann.MatMul(x0, x1, transpose_a=True, data_layout='data_parallel')
    z = lbann.L2Norm2(y)
    obj.append(z)
    metrics.append(lbann.Metric(z, name='TN GEMM'))

    # NumPy implementation
    vals = []
    for i in range(num_samples()):
        x = get_sample(i).astype(np.float64)
        x0 = x[:_N*_m*_k].reshape([_N,_k,_m])
        x1 = x[_N*_m*_k:].reshape([_N,_k,_n])
        y = np.matmul(x0.transpose((0,2,1)), x1)
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
    # NT GEMM
    # ------------------------------------------

    # LBANN implementation
    x0 = lbann.Reshape(x0_lbann, dims=tools.str_list([_N, _m, _k]))
    x1 = lbann.Reshape(x1_lbann, dims=tools.str_list([_N, _n, _k]))
    y = lbann.MatMul(x0, x1, transpose_b=True, data_layout='data_parallel')
    z = lbann.L2Norm2(y)
    obj.append(z)
    metrics.append(lbann.Metric(z, name='NT GEMM'))

    # NumPy implementation
    vals = []
    for i in range(num_samples()):
        x = get_sample(i).astype(np.float64)
        x0 = x[:_N*_m*_k].reshape([_N, _m,_k])
        x1 = x[_N*_m*_k:].reshape([_N, _n,_k])
        x1 = np.transpose(x1, (0, 2, 1))
        y = np.matmul(x0, x1)
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
    # TT GEMM
    # ------------------------------------------

    # LBANN implementation
    x0 = lbann.Reshape(x0_lbann, dims=tools.str_list([_N, _k, _m]))
    x1 = lbann.Reshape(x1_lbann, dims=tools.str_list([_N, _n, _k]))
    y = lbann.MatMul(x0, x1, transpose_a=True, transpose_b=True,
                     data_layout='data_parallel')
    z = lbann.L2Norm2(y)
    obj.append(z)
    metrics.append(lbann.Metric(z, name='TT GEMM'))

    # NumPy implementation
    vals = []
    for i in range(num_samples()):
        x = get_sample(i).astype(np.float64)
        x0 = x[:(_N*_m*_k)].reshape([_N,_k,_m]) 
        x1 = x[_N*_m*_k:].reshape([_N,_n,_k])
        y = np.matmul(x0.transpose((0,2,1)), x1.transpose((0,2,1)))
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
for test in tools.create_tests(setup_experiment, __file__):
    globals()[test.__name__] = test
