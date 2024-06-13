import functools
import math
import operator
import os
import os.path
import sys
import numpy as np
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
_num_samples = 64
_sample_dims = [6,11,7]
_sample_dims_3d = [2,3,11,7]
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
    mini_batch_size = num_samples() // 2
    trainer = lbann.Trainer(mini_batch_size)
    model = construct_model(lbann)
    data_reader = construct_data_reader(lbann)
    optimizer = lbann.NoOptimizer()
    return trainer, model, data_reader, optimizer, None # Don't request any specific number of nodes



upsample_configs = []

# 3x3 upsampling
for mode in ['nearest']:
    upsample_configs.append({
        "name": "3x3 {} upsample".format(mode),
        "scale_factors": (3, 3),
        "upsample_mode": mode
    })

# 2x4 upsampling
for mode in ['nearest']:
    upsample_configs.append({
        "name": "2x4 {} upsample".format(mode),
        "scale_factors": (2, 4),
        "upsample_mode": mode
    })

# 2x2x2 3D upsampling
for mode in ['nearest']:
    upsample_configs.append({
        "name": "2x2x2 {} upsample".format(mode),
        "scale_factors": (2, 2, 2),
        "upsample_mode": mode
    })

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
    # Upsample
    # ------------------------------------------

    for u in upsample_configs:
        uname = u["name"].split(" ")[0]
        num_dims = len(u['scale_factors'])

        # Apply upsampling
        x = x_lbann
        if len(u["scale_factors"]) == 3:
            x = lbann.Reshape(x, dims=_sample_dims_3d)
        x = lbann.Identity(x, name=f'in_{uname}')

        # Convolution settings
        kernel_dims = [_sample_dims[0] if num_dims == 2 else _sample_dims_3d[0]]*2 + [1]*num_dims
        strides = [1]*num_dims
        pads = [0]*num_dims
        dilations = [1]*num_dims

        kernel = np.zeros(kernel_dims)
        for i in range(len(kernel)):
            if num_dims == 2:
                kernel[i,i,0,0] = 1
            else:
                kernel[i,i,0,0,0] = 1

        # Apply convolution
        kernel_weights = lbann.Weights(
            optimizer=lbann.SGD(),
            initializer=lbann.ValueInitializer(values=np.nditer(kernel)),
            name='kernel1_{}'.format(uname)
        )

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
                                  4),
                              name=f'conv1_{uname}')

        y = lbann.Upsample(x,
                           num_dims=len(u["scale_factors"]),
                           has_vectors=True,
                           scale_factors=u["scale_factors"],
                           upsample_mode=u['upsample_mode'],
                           parallel_strategy=create_parallel_strategy(4),
                           name=f'upsample_{uname}')
        

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
            name='kernel2_{}'.format(uname)
        )

        y = lbann.Convolution(y,
                              weights=(kernel_weights, ),
                              num_dims=num_dims,
                              out_channels=kernel_dims[0],
                              kernel_size=kernel_dims[2:],
                              stride=strides,
                              padding=pads,
                              dilation=dilations,
                              has_bias=False,
                              parallel_strategy=create_parallel_strategy(
                                  4),
                              name=f'conv2_{uname}')
        
        y = lbann.Identity(y, name=f'out_{uname}')
        z = lbann.L2Norm2(y)

        obj.append(z)

        # Save the inputs and outputs to check later.
        callbacks.append(lbann.CallbackDumpOutputs(layers=f'in_{uname} out_{uname}', directory='outputs'))

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
        for u in upsample_configs:
            uname = u["name"].split(" ")[0]
            in_data = np.loadtxt(
                os.path.join(
                    experiment_output['work_dir'],
                    'outputs', 'trainer0', 'model0',
                    f'sgd.testing.epoch.0.step.0_in_{uname}_output0.csv'
                ),
                delimiter=','
            )
            out_data = np.loadtxt(
                os.path.join(
                    experiment_output['work_dir'],
                    'outputs', 'trainer0', 'model0',
                    f'sgd.testing.epoch.0.step.0_out_{uname}_output0.csv'
                ),
                delimiter=','
            )
            
            ndims = len(u['scale_factors'])
            upsampled_data = in_data.copy().reshape(
                [-1] + (_sample_dims if ndims == 2 else _sample_dims_3d)
            )
            for i, scale_fac in enumerate(u['scale_factors']):
                if u['upsample_mode'] == 'nearest':
                    upsampled_data = upsampled_data.repeat(scale_fac, axis=i+2)

            assert np.allclose(upsampled_data.ravel(), out_data.ravel())

    # Return test function from factory function
    func.__name__ = test_name
    return func

# Runtime parameters/arguments
environment = lbann.contrib.args.get_distconv_environment()
environment['LBANN_KEEP_ERROR_SIGNALS'] = 1

# Create test functions that can interact with PyTest
# Note: Create test name by removing ".py" from file name
_test_name = os.path.splitext(os.path.basename(current_file))[0]
for _test_func in tools.create_tests(setup_experiment, _test_name,
                                     skip_clusters=["catalyst"],
                                     environment=environment):
    globals()[_test_func.__name__] = augment_test_func(_test_func)