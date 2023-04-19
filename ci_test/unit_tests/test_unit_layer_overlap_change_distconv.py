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
_num_samples = 1
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
    
    mini_batch_size = num_samples() #// 2
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
    x_lbann = lbann.Reshape(lbann.Input(data_field='samples'),
                                dims=_sample_dims)

    # Objects for LBANN model
    obj = []
    metrics = []
    callbacks = []

    # ------------------------------------------
    # Conv -> Identity (Overlap Change) -> Conv
    # ------------------------------------------
    # Two distconv enabled convolutions whose weights produce the identity
    # function. Each convolution has a different kernel size and therefore
    # overlap. An identity layer is placed between the two convolutions to
    # change the overlap.

    num_height_groups = tools.gpus_per_node(lbann)
    if num_height_groups == 0:
        e = 'this test requires GPUs.'
        print('Skip - ' + e)
        pytest.skip(e)

    for num_dims in [2, 3]:
        # Convolution settings
        kernel_dims = [_sample_dims[0] if num_dims == 2 else _sample_dims_3d[0]]*2 + [3]*num_dims
        strides = [1]*num_dims
        pads = [1]*num_dims
        dilations = [1]*num_dims

        kernel = np.zeros(kernel_dims)
        for i in range(len(kernel)):
            if num_dims == 2:
                kernel[i,i,1,1] = 1
            else:
                kernel[i,i,1,1,1] = 1

        # Apply convolution
        kernel_weights = lbann.Weights(
            optimizer=lbann.SGD(),
            initializer=lbann.ValueInitializer(values=np.nditer(kernel)),
            name='kernel1_{}d'.format(num_dims)
        )
        x = x_lbann
        if num_dims == 3:
            x = lbann.Reshape(x, dims=_sample_dims_3d)
        x = lbann.Identity(x, name=f'in_{num_dims}D')

        x = lbann.Convolution(x,
                              weights=(kernel_weights, ),
                              num_dims=num_dims,
                              out_channels=kernel_dims[0],
                              kernel_size=kernel_dims[2:],
                              stride=strides,
                              padding=pads,
                              dilation=dilations,
                              has_bias=False,
                              parallel_strategy=create_parallel_strategy(
                                  num_height_groups),
                              name=f'conv1_{num_dims}D')
        

        # Convolution settings
        kernel_dims = [_sample_dims[0] if num_dims == 2 else _sample_dims_3d[0]]*2 + [5]*num_dims
        strides = [1]*num_dims
        pads = [2]*num_dims
        dilations = [1]*num_dims

        kernel = np.zeros(kernel_dims)
        for i in range(len(kernel)):
            if num_dims == 2:
                kernel[i,i,2,2] = 1
            else:
                kernel[i,i,2,2,2] = 1

        # Apply convolution
        kernel_weights = lbann.Weights(
            optimizer=lbann.SGD(),
            initializer=lbann.ValueInitializer(values=np.nditer(kernel)),
            name='kernel2_{}d'.format(num_dims)
        )

        x = lbann.Identity(x, parallel_strategy=create_parallel_strategy(
                                  num_height_groups), name=f'ident_{num_dims}D')

        x = lbann.Convolution(x,
                              weights=(kernel_weights, ),
                              num_dims=num_dims,
                              out_channels=kernel_dims[0],
                              kernel_size=kernel_dims[2:],
                              stride=strides,
                              padding=pads,
                              dilation=dilations,
                              has_bias=False,
                              parallel_strategy=create_parallel_strategy(
                                  num_height_groups),
                              name=f'conv2_{num_dims}D')
        
        y = lbann.Identity(x, name=f'out_{num_dims}D')
        z = lbann.L2Norm2(y)
        obj.append(z)

        # Save the inputs and outputs to check later.
        callbacks.append(lbann.CallbackDumpOutputs(layers=f'in_{num_dims}D out_{num_dims}D', directory='outputs'))

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

def augment_test_func(test_func):
    """Augment test function to parse log files.

    `tools.create_tests` creates functions that run an LBANN
    experiment. This function creates augmented functions that parse
    the log files after LBANN finishes running, e.g. to check metrics
    or runtimes.

    Note: The naive approach is to define the augmented test functions
    in a loop. However, Python closures are late binding. In other
    words, the function would be overwritten every time we define it.
    We get around this overwriting problem by defining the augmented
    function in the local scope of another function.

    Args:
        test_func (function): Test function created by
            `tools.create_tests`.

    Returns:
        function: Test that can interact with PyTest.

    """
    test_name = test_func.__name__

    # Define test function
    def func(cluster, dirname, weekly):

        # Run LBANN experiment
        experiment_output = test_func(cluster, dirname, weekly)

        # Check that the output values are close to the input values.
        for num_dims in [2, 3]:
            in_data = np.loadtxt(
                os.path.join(
                    experiment_output['work_dir'],
                    'outputs', 'trainer0', 'model0',
                    f'sgd.testing.epoch.0.step.0_in_{num_dims}D_output0.csv'
                ),
                delimiter=','
            )
            out_data = np.loadtxt(
                os.path.join(
                    experiment_output['work_dir'],
                    'outputs', 'trainer0', 'model0',
                    f'sgd.testing.epoch.0.step.0_out_{num_dims}D_output0.csv'
                ),
                delimiter=','
            )

            assert np.allclose(in_data, out_data, rtol=1e-3)

    # Return test function from factory function
    func.__name__ = test_name
    return func

# Create test functions that can interact with PyTest
# Note: Create test name by removing ".py" from file name
_test_name = os.path.splitext(os.path.basename(current_file))[0]
for _test_func in tools.create_tests(setup_experiment, _test_name,
                               environment=tools.get_distconv_environment()):
    globals()[_test_func.__name__] = augment_test_func(_test_func)
