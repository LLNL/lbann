"""Test to check polynomial decay learning rate schedule.

LBANN is run with the polynomial learning rate schedule and the log
files are post-processed to make sure that the correct learning rate
values are used.

"""
import os
import os.path
import random
import re
import sys

# Bamboo utilities
current_file = os.path.realpath(__file__)
current_dir = os.path.dirname(current_file)
sys.path.insert(0, os.path.join(os.path.dirname(current_dir), 'common_python'))
import tools

# ==============================================
# Learning rate schedule parameters
# ==============================================

lr_power = 0.8
lr_num_epochs = 5
lr_start = 1
lr_end = 0.1

# ==============================================
# Setup LBANN experiment
# ==============================================

def setup_experiment(lbann, weekly):
    """Construct LBANN experiment.

    Args:
        lbann (module): Module for LBANN Python frontend

    """
    mini_batch_size = 1
    trainer = lbann.Trainer(mini_batch_size)
    model = construct_model(lbann)
    data_reader = construct_data_reader(lbann)
    optimizer = lbann.SGD(learn_rate=lr_start)
    return trainer, model, data_reader, optimizer, None # Don't request any specific number of nodes

def construct_model(lbann):
    """Construct LBANN model.

    Args:
        lbann (module): Module for LBANN Python frontend

    """

    # Layer graph
    x = lbann.Input(data_field='samples')
    x = lbann.FullyConnected(x, num_neurons=1)

    # Model objects
    metrics = []
    callbacks = [
        lbann.CallbackPolyLearningRate(
            power=lr_power,
            num_epochs=lr_num_epochs,
            end_lr=lr_end,
        ),
    ]

    # Construct model
    return lbann.Model(lr_num_epochs+2,
                       layers=x,
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
    _reader = message.reader.add()
    _reader.name = 'synthetic'
    _reader.role = 'train'
    _reader.num_samples = 2
    _reader.synth_dimensions = '1'
    _reader.fraction_of_data_to_use = 1.0
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

        # Parse LBANN log file
        lr_list = []
        log_file = experiment_output['stdout_log_file']
        with open(log_file) as f:
            for line in f:
                match = re.search(
                    'changing global learning rate to ([0-9.]+)',
                    line)
                if match:
                    lr_list.append(float(match.group(1)))

        # Make sure file has been parsed correctly
        assert len(lr_list) == lr_num_epochs, \
            f'Error parsing {log_file} ' \
            f'(expected {lr_num_epochs} learning rates, ' \
            f'but found {len(lr_list)})'

        # Make sure learning rates match expected values
        tol = 1e-5
        for epoch in range(lr_num_epochs):
            lr = lr_list[epoch]
            scale = (1 - (epoch+1)/lr_num_epochs) ** lr_power
            expected_lr = (lr_start - lr_end) * scale + lr_end
            assert expected_lr-tol < lr < expected_lr+tol, \
                f'Incorrect learning rate at epoch {epoch}' \
                f'(expected {expected_lr}, but found {lr})'

    # Return test function from factory function
    func.__name__ = test_name
    return func

# Create test functions that can interact with PyTest
for _test_func in tools.create_tests(setup_experiment, __file__,):
    globals()[_test_func.__name__] = augment_test_func(_test_func)
