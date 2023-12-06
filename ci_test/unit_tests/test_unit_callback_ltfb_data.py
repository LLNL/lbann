"""Test to check data ingestion with LTFB.

LTFB requires switching between training and validation data sets in
the middle of a training epoch. We perform metric checking to make
sure that no samples are skipped or duplicated during these
transitions.

"""
import os
import os.path
import random
import sys

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

# Parameters
_mini_batch_size = 7

# Sample access functions
def get_train_sample(index):
    return (index,)
def get_val_sample(index):
    return (10+index,)
def get_test_sample(index):
    return (100+index,)
def num_train_samples():
    return 31
def num_val_samples():
    return 13
def num_test_samples():
    return 42
def sample_dims():
    return (1,)

# ==============================================
# Setup LBANN experiment
# ==============================================

def setup_experiment(lbann, weekly):
    """Construct LBANN experiment.

    Args:
        lbann (module): Module for LBANN Python frontend

    """
    trainer = lbann.Trainer(_mini_batch_size)
    model = construct_model(lbann)
    data_reader = construct_data_reader(lbann)
    optimizer = lbann.NoOptimizer()
    return trainer, model, data_reader, optimizer, None # Don't request any specific number of nodes

def construct_model(lbann):
    """Construct LBANN model.

    Args:
        lbann (module): Module for LBANN Python frontend

    """

    # Layer graph
    step_id = lbann.Input(data_field='samples')
    for l in lbann.traverse_layer_graph(step_id):
        l.device = 'CPU'

    # LTFB
    ltfb_interval = 3
    metrics = [lbann.Metric(step_id, name='step id')]
    callbacks = [
        lbann.CallbackPrint(),
        lbann.CallbackLTFB(
            batch_interval=ltfb_interval,
            metric=metrics[-1].name,
        ),
    ]

    # Metric checking
    # Note: sum(i) = (n-1) / 2
    train_val = (num_train_samples() - 1) / 2
    val_val = 10 + (num_val_samples() - 1) / 2
    test_val = 100 + (num_test_samples() - 1) / 2
    tol = 1e-4
    callbacks.extend([
        lbann.CallbackCheckMetric(
            metric=metrics[-1].name,
            lower_bound=train_val-tol,
            upper_bound=train_val+tol,
            error_on_failure=True,
            execution_modes='train'),
        lbann.CallbackCheckMetric(
            metric=metrics[-1].name,
            lower_bound=val_val-tol,
            upper_bound=val_val+tol,
            error_on_failure=True,
            execution_modes='validation'),
        lbann.CallbackCheckMetric(
            metric=metrics[-1].name,
            lower_bound=test_val-tol,
            upper_bound=test_val+tol,
            error_on_failure=True,
            execution_modes='test'),
    ])

    # Choose number of epochs so that LTFB round and epoch line up
    num_epochs = 2 * ltfb_interval

    # Construct model
    return lbann.Model(num_epochs,
                       layers=lbann.traverse_layer_graph(step_id),
                       metrics=metrics,
                       callbacks=callbacks)

def construct_data_reader(lbann):
    """Construct Protobuf message for Python data reader.

    The Python data reader will import the current Python file to
    access the sample access functions.

    Args:
        lbann (module): Module for LBANN Python frontend

    """

    message = lbann.reader_pb2.DataReader()
    message.reader.extend([
        tools.create_python_data_reader(
            lbann,
            current_file,
            'get_train_sample',
            'num_train_samples',
            'sample_dims',
            'train',
        ),
        tools.create_python_data_reader(
            lbann,
            current_file,
            'get_val_sample',
            'num_val_samples',
            'sample_dims',
            'validate',
        ),
        tools.create_python_data_reader(
            lbann,
            current_file,
            'get_val_sample',
            'num_val_samples',
            'sample_dims',
            'tournament',
        ),
        tools.create_python_data_reader(
            lbann,
            current_file,
            'get_test_sample',
            'num_test_samples',
            'sample_dims',
            'test',
        ),
    ])
    return message

# ==============================================
# Setup PyTest
# ==============================================

# Create test functions that can interact with PyTest
for _test_func in tools.create_tests(setup_experiment,
                               __file__,
                               nodes=2,
                               time_limit=3,
                               lbann_args='--procs_per_trainer=2'):
    globals()[_test_func.__name__] = _test_func
