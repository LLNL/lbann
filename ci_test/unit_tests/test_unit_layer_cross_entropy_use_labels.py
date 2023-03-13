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
# Note: The error bounds for gradient checking assume that the fourth
# derivative of the objective function is ~1. However, given our loss
# function:
#   L = ( -xhat * log(x) )^2
#   L'''' = O( xhat^2 * log(x) / x^4 )
# We have x >= 0.25 to make sure the fourth derivative does not get
# too big and mess up the error bounds.
np.random.seed(202303102)
SAMPLE_SPATIAL_SIZE = 4**3
NUM_CHANNELS = 3
NUM_SAMPLES = 23
_samples = np.random.uniform(low=0.25,
                             high=1,
                             size=(NUM_SAMPLES, NUM_CHANNELS, SAMPLE_SPATIAL_SIZE)).astype(np.float32)
_labels = np.random.randint(NUM_CHANNELS, size=(NUM_SAMPLES, 1, SAMPLE_SPATIAL_SIZE)).astype(np.float32)


# Sample access functions
def get_sample(index):
    return np.concatenate([_samples[index].reshape(-1),
                           _labels[index].reshape(-1)])
def num_samples():
    return NUM_SAMPLES
def sample_dims():
    return (_samples.shape[-1] + _labels.shape[-1],)

# ==============================================
# NumPy cross entropy
# ==============================================

def numpy_cross_entropy(x, xhat):
    """Cross entropy between a distribution and ground truth labels, 
       computed with NumPy

    The computation is performed with 64-bit floats.

    Args:
        x: Estimated distribution
        xhat: Labels of the true distribution

    """
    if x.dtype is not np.float64:
        x = x.astype(np.float64)
    if xhat.dtype is not np.float64:
        xhat = xhat.astype(np.float64)
    
    x = x.flatten()
    xhat = xhat.flatten()
    loss = 0
    for j in range(SAMPLE_SPATIAL_SIZE):
        correct_channel = xhat[j]
        offset = int((correct_channel * SAMPLE_SPATIAL_SIZE) + j)
        loss += np.log(x[offset])
    return -loss 

# ==============================================
# Setup LBANN experiment
# ==============================================

def setup_experiment(lbann, weekly):
    """Construct LBANN experiment.

    Args:
        lbann (module): Module for LBANN Python frontend

    """
    mini_batch_size = NUM_SAMPLES() // 2
    trainer = lbann.Trainer(mini_batch_size)
    model = construct_model(lbann)
    data_reader = construct_data_reader(lbann)
    optimizer = lbann.NoOptimizer()
    return trainer, model, data_reader, optimizer, None # Don't request any specific number of nodes

def construct_model(lbann):
    """Construct LBANN model.

    Args:
        lbann (module): Module for LBANN Python frontend

    """

    # Input data
    # Note: Sum with weights layers so that gradient checking will
    # verify that error signals are correct.
    slice_size = NUM_CHANNELS * SAMPLE_SPATIAL_SIZE
    data_size = slice_size + SAMPLE_SPATIAL_SIZE
    x0_weights = lbann.Weights(optimizer=lbann.SGD(),
                               initializer=lbann.ConstantInitializer(value=0.0),
                               name='input0_weights')
    # x1_weights = lbann.Weights(optimizer=lbann.SGD(),
    #                            initializer=lbann.ConstantInitializer(value=0.0),
    #                            name='input1_weights')
    x_slice = lbann.Slice(lbann.Input(data_field='samples'),
                          slice_points=[0, slice_size, data_size])
    x0 = lbann.Sum(x_slice,
                   lbann.WeightsLayer(weights=x0_weights, dims=[slice_size]))
    # x1 = lbann.Sum(x_slice,
    #                lbann.WeightsLayer(weights=x1_weights, dims=[SAMPLE_SPATIAL_SIZE]))
    # Don't do gradient check for the labels
    x1 = lbann.Reshape(lbann.Identity(x_slice), dims=[SAMPLE_SPATIAL_SIZE])
    x0_lbann = x0
    x1_lbann = x1

    # Objects for LBANN model
    obj = []
    metrics = []
    callbacks = []

    # ------------------------------------------
    # Data-parallel layout (2D)
    # ------------------------------------------

    sample_shape_2d = [NUM_CHANNELS,
                       int(np.sqrt(SAMPLE_SPATIAL_SIZE)),
                       int(np.sqrt(SAMPLE_SPATIAL_SIZE))]
    
    label_shape_2d = [1,
                      int(np.sqrt(SAMPLE_SPATIAL_SIZE)),
                      int(np.sqrt(SAMPLE_SPATIAL_SIZE))]
    
    # LBANN implementation
    x0 = x0_lbann
    x1 = x1_lbann

    x0 = lbann.reshape(x0, dims=sample_shape_2d)
    x1 = lbann.reshape(x1, dims=label_shape_2d)

    y = lbann.CrossEntropy(x0, x1, data_layout='data_parallel', use_labels=True)
    z = lbann.L2Norm2(y)
    obj.append(z)
    metrics.append(lbann.Metric(z, name='data-parallel layout'))

    # NumPy implementation
    vals = []
    for i in range(num_samples()):
        x = get_sample(i).astype(np.float64)
        x0 = x[:slice_size]
        x1 = x[slice_size:]
        y = numpy_cross_entropy(x0, x1)
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
    # Data-parallel layout (3D)
    # ------------------------------------------

    # We expect the same exact results as the 2D case.
    # This unit test is to make sure the usually expected
    # sample dimensions are well supported and the dimensionality
    # check is working correctly. 

    sample_shape_3d = [NUM_CHANNELS,
                       int(np.cbrt(SAMPLE_SPATIAL_SIZE)),
                       int(np.cbrt(SAMPLE_SPATIAL_SIZE)),
                       int(np.cbrt(SAMPLE_SPATIAL_SIZE))]
    
    label_shape_3d = [1,
                      int(np.cbrt(SAMPLE_SPATIAL_SIZE)),
                      int(np.cbrt(SAMPLE_SPATIAL_SIZE)),
                      int(np.cbrt(SAMPLE_SPATIAL_SIZE))]
    # LBANN implementation
    x0 = x0_lbann
    x1 = x1_lbann

    x0 = lbann.reshape(x0, dims=sample_shape_3d)
    x1 = lbann.reshape(x1, dims=label_shape_3d)

    y = lbann.CrossEntropy(x0, x1, data_layout='data_parallel', use_labels=True)
    z = lbann.L2Norm2(y)
    obj.append(z)
    metrics.append(lbann.Metric(z, name='data-parallel layout'))

    # NumPy implementation
    vals = []
    for i in range(num_samples()):
        x = get_sample(i).astype(np.float64)
        x0 = x[:slice_size]
        x1 = x[slice_size:]
        y = -np.inner(x1, np.log(x0))
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
# Note: Create test name by removing ".py" from file name
_test_name = os.path.splitext(os.path.basename(current_file))[0]
for _test_func in tools.create_tests(setup_experiment, _test_name):
    globals()[_test_func.__name__] = _test_func
