import operator
import os
import os.path
import sys
import numpy as np
import functools

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
num_rows = 5
num_columns = 20
input_size = num_rows * num_columns
output_size = 15
seed = 202101280

# Sample access functions
def get_sample(index):
    np.random.seed(seed+index)
    values = [np.random.normal() for _ in range(input_size)]
    indices = [
        np.random.uniform(-1, input_size+1)
        for _ in range(output_size)
    ]
    return values + indices
def num_samples():
    return 25
def sample_dims():
    return (input_size+output_size,)

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
    # Note: Sum with a weights layer so that gradient checking will
    # verify that error signals are correct.
    x = lbann.Identity(lbann.Input())
    x_slice = lbann.Slice(
        x,
        slice_points=tools.str_list([0,input_size,input_size+output_size]),
    )
    x0_weights = lbann.Weights(
        optimizer=lbann.SGD(),
        initializer=lbann.ConstantInitializer(value=0.0),
        name='input_weights',
    )
    x0 = lbann.Sum(
        lbann.Identity(x_slice),
        lbann.WeightsLayer(weights=x0_weights, dims=tools.str_list(input_size)),
    )
    x1 = lbann.Identity(x_slice)

    x0_lbann = x0
    x1_lbann = x1

    # Apply gather
    y0 = lbann.Gather(x0, x1)
    y1 = lbann.Concatenation([
        lbann.Constant(value=i+1, num_neurons='1')
        for i in range(output_size)
    ])
    y = lbann.Multiply(y0, y1)
    z = lbann.L2Norm2(y)

    # Objects for LBANN model
    metrics = []
    callbacks = []
    objs = []

    metrics.append(lbann.Metric(z, name='1D_obj'))
    objs.append(lbann.ObjectiveFunction(z))
    # Compute expected metric value
    vals = []
    for i in range(num_samples()):
        sample = get_sample(i)
        x0 = sample[:input_size]
        x1 = sample[input_size:]
        y0 = np.zeros(output_size)
        for i in range(output_size):
            if 0 <= x1[i] < input_size:
                y0[i] = x0[int(x1[i])]
        z = 0
        for i in range(output_size):
            z += ((i+1)*y0[i]) ** 2
        vals.append(z)
    val = np.mean(vals)
    tol = 8 * val * np.finfo(np.float32).eps
    callbacks.append(lbann.CallbackCheckMetric(
        metric=metrics[-1].name,
        lower_bound=val-tol,
        upper_bound=val+tol,
        error_on_failure=True,
        execution_modes='test'))

    ######################################################################
    #
    #          2D Values , 1D Input 
    #
    ######################################################################

    x0 = lbann.Reshape(x0_lbann, dims=tools.str_list([num_rows, num_columns]))
    
    x1 = lbann.Identity(x1_lbann, name="Indices_2D")

    y0 = lbann.Gather(x0,x1, name="Gather_2D")

    y1 = lbann.Concatenation([
        lbann.Constant(value=i+1, num_neurons='1')
        for i in range(num_rows * output_size)])

    y0 = lbann.Reshape(y0, dims=tools.str_list([num_rows * output_size]))
    y1 = lbann.Reshape(y1, dims=tools.str_list([num_rows * output_size]))

    y = lbann.Multiply(y0, y1)

    z = lbann.L2Norm2(y)

    objs.append(z)
    metrics.append(lbann.Metric(z, name="2D_obj"))

    vals = []
    for i in range(num_samples()):
        sample = get_sample(i)
        x0 = np.array(sample[:input_size]).reshape((num_rows, num_columns))
        x1 = sample[input_size:input_size + output_size]

        y0 = np.zeros((num_rows, output_size))

        for i in range(num_rows):
            for j in range(output_size):
                if 0 <= x1[j] <= num_columns:
                    y0[i][j] = x0[i][int(x1[j])]
        z = 0
        for i in range(num_rows * output_size):
            z += ((i+1)*y0.flatten()[i])**2
        vals.append(z)

    val = np.mean(vals)
    tol = 8 * val * np.finfo(np.float32).eps
    callbacks.append(lbann.CallbackCheckMetric(
        metric=metrics[-1].name,
        lower_bound=val-tol,
        upper_bound=val+tol,
        error_on_failure=True,
        execution_modes='test'))

    # Gradient checking
    callbacks.append(lbann.CallbackCheckGradients(error_on_failure=True))

    # Construct model
    num_epochs = 0
    layers = list(lbann.traverse_layer_graph(x))
    return lbann.Model(num_epochs,
                       layers=layers,
                       objective_function=objs,
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
