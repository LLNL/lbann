import functools
import math
import operator
import os
import os.path
import sys
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

def make_random_array(shape, seed):
    """Hacked function to generate a random array.

    NumPy's RNG produces different values with different NumPy
    versions. This function is helpful when array values must be
    identical across all runs, e.g. when checking against precomputed
    metric values.

    Args:
        shape (Iterable of int): Array dimensions
        seed (int): Parameter for RNG. Must be non-zero.
    Returns:
        numpy.ndarray: Array of `np.float32`. Values will be in
            [-0.5,0.5).

    """
    size = functools.reduce(operator.mul, shape)
    eps = np.finfo(np.float32).eps
    x = (seed / np.linspace(math.sqrt(eps), 0.1, size)) % 1 - 0.5
    return x.reshape(shape).astype(np.float32)

# Data
_num_samples = 8
_sample_dims = [64,16,16]
_sample_dims_3d = [4,16,16,16]
_sample_size = functools.reduce(operator.mul, _sample_dims)
_samples = make_random_array([_num_samples] + _sample_dims, 7)

# Sample access functions
def get_sample(index):
    return _samples[index,:].reshape(-1)
def num_samples():
    return _num_samples
def sample_dims():
    return (_sample_size,)

# ==============================================
# PyTorch pooling
# ==============================================

def pytorch_pooling(data,
                    kernel_dims,
                    pool_mode,
                    stride=1,
                    padding=0):
    """Wrapper around PyTorch pooling.

    Input and output data are NumPy arrays.

    """

    # Convert input data to PyTorch tensors with 64-bit floats
    import torch
    import torch.nn.functional
    if type(data) is np.ndarray:
        data = torch.from_numpy(data)
    if data.dtype is not torch.float64:
        data = data.astype(torch.float64)

    # Perform pooling with PyTorch
    if len(kernel_dims) not in [1, 2, 3]:
        raise ValueError('PyTorch only supports 1D, 2D, and 3D pooling')

    func_name = "{}_pool{}d".format(
        {"average": "avg", "max": "max"}[pool_mode],
        len(kernel_dims),
    )
    output = getattr(torch.nn.functional, func_name)(
        data, kernel_dims, stride, padding,
    )

    # Return output as NumPy array
    return output.numpy()

# ==============================================
# Setup LBANN experiment
# ==============================================

def setup_experiment(lbann, weekly):
    """Construct LBANN experiment.

    Args:
        lbann (module): Module for LBANN Python frontend

    """
    if not lbann.has_feature('DISTCONV'):
        message = f'{os.path.basename(__file__)} requires DISTCONV'
        print('Skip - ' + message)
        pytest.skip(message)
    
    mini_batch_size = num_samples() // 2
    trainer = lbann.Trainer(mini_batch_size=mini_batch_size)
    model = construct_model(lbann)
    data_reader = construct_data_reader(lbann)
    optimizer = lbann.NoOptimizer()
    return trainer, model, data_reader, optimizer, None # Don't request any specific number of nodes

def create_parallel_strategy(num_height_groups):
    return {"height_groups": num_height_groups}

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
    x = lbann.Sum(lbann.Reshape(lbann.Input(data_field='samples'),
                                dims=_sample_dims),
                  lbann.WeightsLayer(weights=x_weights,
                                     dims=_sample_dims))
    x_lbann = x

    # Objects for LBANN model
    obj = []
    metrics = []
    callbacks = []

    # ------------------------------------------
    # Pooling
    # ------------------------------------------

    num_height_groups = tools.gpus_per_node(lbann)
    if num_height_groups == 0:
        e = 'this test requires GPUs.'
        print('Skip - ' + e)
        pytest.skip(e)

    pool_configs = []

    # 3x3 pooling with same padding
    for mode, val in [
            ("average", 700.1066377082393), # _num_samples=8
            ("max", 1255.4813455546334), # _num_samples=8
            # ("average", 830.2573008820838), # _num_samples=23
            # ("max", 1167.667676299899), # _num_samples=23
    ]:
        pool_configs.append({
            "name": "3x3 {} pooling".format(mode),
            "kernel_dims": (3, 3),
            "strides": (1, 1),
            "pads": (0, 0),
            "pool_mode": mode,
            "val": val,
        })

    # 2x2 strided pooling
    for mode, val in [
            ("average", 263.76437243059104), # _num_samples=23
            ("max", 358.66104389177207), # _num_samples=23
            # ("average", 293.61402789516825), # _num_samples=23
            # ("max", 351.4916288366334), # _num_samples=23
    ]:
        pool_configs.append({
            "name": "2x2 {} pooling".format(mode),
            "kernel_dims": (2, 2),
            "strides": (2, 2),
            "pads": (0, 0),
            "pool_mode": mode,
            "val": val,
        })

    # 2x2x2 3D pooling
    for mode, val in [
            ("average", 59.3851451701403), # _num_samples=8
            ("max", 216.75871475407558), # _num_samples=8
            # ("average", 89.61246528381926), # _num_samples=23
            # ("max", 198.65624293856985), # _num_samples=23
    ]:
        pool_configs.append({
            "name": "2x2x2 {} pooling".format(mode),
            "kernel_dims": (2, 2, 2),
            "strides": (2, 2, 2),
            "pads": (0, 0, 0),
            "pool_mode": mode,
            "val": val,
        })

    for p in pool_configs:
        # Apply pooling
        x = x_lbann
        if len(p["kernel_dims"]) == 3:
            x = lbann.Reshape(x, dims=_sample_dims_3d)

        y = lbann.Pooling(x,
                          num_dims=len(p["kernel_dims"]),
                          has_vectors=True,
                          pool_dims=p["kernel_dims"],
                          pool_strides=p["strides"],
                          pool_pads=p["pads"],
                          pool_mode=p["pool_mode"],
                          parallel_strategy=create_parallel_strategy(
                              num_height_groups))
        z = lbann.L2Norm2(y)

        # Since max pooling is not differentiable, we only use average pooling.
        if p["pool_mode"] == "average":
            obj.append(z)

        metrics.append(lbann.Metric(z, name=p["name"]))

        # PyTorch implementation
        try:
            x = _samples
            if len(p["kernel_dims"]) == 3:
                x = np.reshape(x, [_num_samples]+_sample_dims_3d)

            y = pytorch_pooling(
                x,
                p["kernel_dims"],
                p["pool_mode"],
                stride=p["strides"],
                padding=p["pads"],
            )
            z = tools.numpy_l2norm2(y) / _num_samples
            val = z
        except:
            # Precomputed value
            val = p["val"]
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
# Note: Create test name by removing ".py" from file name
_test_name = os.path.splitext(os.path.basename(current_file))[0]
for _test_func in tools.create_tests(setup_experiment, _test_name,
                               environment=lbann.contrib.args.get_distconv_environment()):
    globals()[_test_func.__name__] = _test_func
