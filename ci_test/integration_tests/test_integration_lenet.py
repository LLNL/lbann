import functools
import operator
import os
import os.path
import re
import sys
import numpy as np
import google.protobuf.text_format
import warnings
import pytest

# Local files
current_file = os.path.realpath(__file__)
current_dir = os.path.dirname(current_file)
sys.path.insert(0, os.path.join(os.path.dirname(current_dir), 'common_python'))
import tools
import data.mnist

# ==============================================
# Options
# ==============================================

# Training options
num_epochs = 5
mini_batch_size = 64
num_nodes = 2

# Classification accuracy (percent)
expected_train_accuracy_range = (98.75, 99.25)
expected_test_accuracy_range = (98, 99)

# Average mini-batch time (in sec) for each LC system
# Note that run times are with LBANN_DETERMINISTIC set
# Commented out times are prior to thread safe RNGs
expected_mini_batch_times = {
    'pascal':   0.0020, # Changed as of 1/18/22 0.0014, # 0.0013,
    'catalyst': 0.0073, # 0.0055,
    'lassen':   0.0022,
    'ray':      0.0029, # Changed as of 3/22/22 0.0025,
    'corona':   0.0117, # 0.0075,
    'tioga':    0.0020, # BVE dummy value from pascal
}

# ==============================================
# Setup LBANN experiment
# ==============================================

def setup_experiment(lbann, weekly):
    """Construct LBANN experiment.

    Args:
        lbann (module): Module for LBANN Python frontend

    """
    if lbann.has_feature('DETERMINISTIC') and lbann.has_feature('MIOPEN') :
        message = f'{os.path.basename(__file__)} Deterministic max pooling mode not supported in MIOpen'
        print('Skip - ' + message)
        pytest.skip(message)
    trainer = lbann.Trainer(mini_batch_size=mini_batch_size)
    model = construct_model(lbann)

    data_reader = data.mnist.make_data_reader(lbann)
    # No validation set
    data_reader.reader[0].validation_fraction = 0

    optimizer = lbann.SGD(learn_rate=0.01, momentum=0.9)
    return trainer, model, data_reader, optimizer, num_nodes

def construct_model(lbann):
    """Construct LBANN model.

    Args:
        lbann (module): Module for LBANN Python frontend

    """

    # TODO (tym): Figure out how to switch between LBANN builds. See
    # GitHub Issue #1289.
    import lbann.models

    # Layer graph
    images = lbann.Input(data_field='samples')
    labels = lbann.Input(data_field='labels')
    x = lbann.models.LeNet(10)(images)
    probs = lbann.Softmax(x)
    loss = lbann.CrossEntropy(probs, labels)
    acc = lbann.CategoricalAccuracy(probs, labels)

    # Objects for LBANN model
    callbacks = [lbann.CallbackPrint(), lbann.CallbackTimer()]
    metrics = [lbann.Metric(acc, name='accuracy', unit='%')]

    # Construct model
    return lbann.Model(num_epochs,
                       layers=lbann.traverse_layer_graph([images, labels]),
                       objective_function=loss,
                       metrics=metrics,
                       callbacks=callbacks)

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
        train_accuracy = None
        test_accuracy = None
        mini_batch_times = []
        with open(experiment_output['stdout_log_file']) as f:
            for line in f:
                match = re.search('training epoch [0-9]+ accuracy : ([0-9.]+)%', line)
                if match:
                    train_accuracy = float(match.group(1))
                match = re.search('test accuracy : ([0-9.]+)%', line)
                if match:
                    test_accuracy = float(match.group(1))
                match = re.search('training epoch [0-9]+ mini-batch time statistics : ([0-9.]+)s mean', line)
                if match:
                    mini_batch_times.append(float(match.group(1)))

        # Check if training accuracy is within expected range
        assert ((train_accuracy > expected_train_accuracy_range[0]
                 and train_accuracy < expected_train_accuracy_range[1])), \
                f"train accuracy {train_accuracy:.3f} is outside expected range " + \
                f"[{expected_train_accuracy_range[0]:.3f},{expected_train_accuracy_range[1]:.3f}]"

        # Check if testing accuracy is within expected range
        assert ((test_accuracy > expected_test_accuracy_range[0]
                 and test_accuracy < expected_test_accuracy_range[1])), \
                f"test accuracy {test_accuracy:.3f} is outside expected range " + \
                f"[{expected_test_accuracy_range[0]:.3f},{expected_test_accuracy_range[1]:.3f}]"

        # Check if mini-batch time is within expected range
        # Note: Skip first epoch since its runtime is usually an outlier
        mini_batch_times = mini_batch_times[1:]
        mini_batch_time = sum(mini_batch_times) / len(mini_batch_times)
        min_expected_mini_batch_time = 0.75 * expected_mini_batch_times[cluster]
        max_expected_mini_batch_time = 1.25 * expected_mini_batch_times[cluster]
        if (mini_batch_time < min_expected_mini_batch_time or
            mini_batch_time > max_expected_mini_batch_time):
            warnings.warn(f'average mini-batch time {mini_batch_time:.3f} is outside expected range ' +
                          f'[{min_expected_mini_batch_time:.3f}, {max_expected_mini_batch_time:.3f}]', UserWarning)

    # Return test function from factory function
    func.__name__ = test_name
    return func

# Create test functions that can interact with PyTest
for _test_func in tools.create_tests(setup_experiment,
                                     __file__):
    globals()[_test_func.__name__] = augment_test_func(_test_func)
