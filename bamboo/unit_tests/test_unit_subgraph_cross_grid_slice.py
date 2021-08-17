import functools
import operator
import os
import os.path
import sys
import numpy as np

# Bamboo utilities
current_file = os.path.realpath(__file__)
print("Current file:",current_file, __file__)
current_dir = os.path.dirname(current_file)
sys.path.insert(0, os.path.join(os.path.dirname(current_dir), 'common_python'))
import tools

# ==============================================
# Objects for Python data reader
# ==============================================
# Note: The Python data reader imports this file as a module and calls
# the functions below to ingest data.

# Data
np.random.seed(2019102455)
_num_samples = 128
_sample_size = 64
_sample_dims = (2,2,16)
_sample_size = functools.reduce(operator.mul, _sample_dims)
_samples = np.random.normal(size=(_num_samples,_sample_size)).astype(np.float32)

# Sample access functions
def get_sample(index):
    return _samples[index,:]
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
    x = lbann.Sum(lbann.Reshape(lbann.Input(data_field='datum'),
                                dims=tools.str_list(_sample_dims)),
                  lbann.WeightsLayer(weights=x_weights,
                                     dims=tools.str_list(_sample_dims)))
    x_lbann = x

    # Objects for LBANN model
    obj = []
    metrics = []
    callbacks = []

    branches = 2

    # ------------------------------------------
    # Data-parallel layout
    # ------------------------------------------

    # LBANN implementation
    x = x_lbann
    y = lbann.Identity(x, data_layout='data_parallel')

    slice_points = (0, 8, 16)
    x_slice = lbann.Slice(x, axis=2, slice_points=tools.str_list(slice_points),parallel_strategy = {'sub_branch_tag':0,'enable_subgraph':True})

    branch1 = lbann.Identity(x_slice, data_layout='data_parallel',parallel_strategy = {'sub_branch_tag':1,'enable_subgraph':True})
    branch2 = lbann.Identity(x_slice, data_layout='data_parallel',parallel_strategy = {'sub_branch_tag':2,'enable_subgraph':True})

    grid_slice = lbann.Cross_Grid_Sum_Slice([branch1,branch2])

    branch1 = lbann.Identity(grid_slice)
    branch2 = lbann.Identity(grid_slice)

    sum_branch = lbann.Sum([branch1,branch2],parallel_strategy = {'sub_branch_tag':0,'enable_subgraph':True})
    z = lbann.L2Norm2(sum_branch)
    obj.append(z)
    metrics.append(lbann.Metric(z, name='data-parallel layout'))

    # NumPy implementation
    vals = []
    for i in range(num_samples()):
        x = get_sample(i).reshape(_sample_dims).astype(np.float64)
        y = []

        cross_sum = 0
        for j in range(len(slice_points)-1):
            x_slice = x[:,:,slice_points[j]:slice_points[j+1]]

            if(j==0):
                cross_sum = x_slice
            else:
                cross_sum += x_slice

        last_dim = int(_sample_dims[-1]/(branches*branches))
        sum_slices = None
        for j in range(branches):
            if(j==0):
                sum_slices = cross_sum[:,:,:last_dim]
            else:
                sum_slices += cross_sum[:,:,j*last_dim:(j+1)*last_dim]

        z = tools.numpy_l2norm2(sum_slices)
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

    return lbann.Model(num_epochs,subgraph_communication=lbann.SubgraphCommunication.COLL_OPT,
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
