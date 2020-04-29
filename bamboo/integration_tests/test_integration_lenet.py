import functools
import operator
import os
import os.path
import re
import sys
import numpy as np
import google.protobuf.text_format
import pytest

# Local files
current_file = os.path.realpath(__file__)
current_dir = os.path.dirname(current_file)
sys.path.insert(0, os.path.join(os.path.dirname(current_dir), 'common_python'))
import tools

# Applications directories
bamboo_dir = os.path.dirname(current_dir)
lbann_dir = os.path.dirname(bamboo_dir)
sys.path.insert(0, os.path.join(lbann_dir, 'applications', 'vision'))
import data.mnist

# Python directory
sys.path.insert(0, os.path.join(lbann_dir, 'python'))

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
expected_mini_batch_times = {
    'pascal':   0.0012,
    'catalyst': 0.0055,
    'lassen':   0.0020,
    'ray':      0.0025,
    'corona':   0.0075,
}

# ==============================================
# Setup LBANN experiment
# ==============================================

def setup_experiment(lbann):
    """Construct LBANN experiment.

    Args:
        lbann (module): Module for LBANN Python frontend

    """
    trainer = lbann.Trainer()
    model = construct_model(lbann)

    data_reader = data.mnist.make_data_reader(download_data=False)
    # No validation set
    data_reader.reader[0].validation_percent = 0

    optimizer = lbann.SGD(learn_rate=0.01, momentum=0.9)
    return trainer, model, data_reader, optimizer

def construct_model(lbann):
    """Construct LBANN model.

    Args:
        lbann (module): Module for LBANN Python frontend

    """

    # TODO (tym): Figure out how to switch between LBANN builds. See
    # GitHub Issue #1289.
    import lbann.models

    # Layer graph
    input_ = lbann.Input()
    images = lbann.Identity(input_)
    labels = lbann.Identity(input_)
    x = lbann.models.LeNet(10)(images)
    probs = lbann.Softmax(x)
    loss = lbann.CrossEntropy(probs, labels)
    acc = lbann.CategoricalAccuracy(probs, labels)

    # Objects for LBANN model
    callbacks = [lbann.CallbackPrint(), lbann.CallbackTimer()]
    metrics = [lbann.Metric(acc, name='accuracy', unit='%')]

    # Construct model
    return lbann.Model(mini_batch_size,
                       num_epochs,
                       layers=lbann.traverse_layer_graph(input_),
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
    def func(cluster, exes, dirname):

        # Run LBANN experiment
        experiment_output = test_func(cluster, exes, dirname)

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
        assert (expected_train_accuracy_range[0]
                < train_accuracy
                < expected_train_accuracy_range[1]), \
                'train accuracy is outside expected range'

        # Check if testing accuracy is within expected range
        assert (expected_test_accuracy_range[0]
                < test_accuracy
                < expected_test_accuracy_range[1]), \
                'test accuracy is outside expected range'

        # Check if mini-batch time is within expected range
        # Note: Skip first epoch since its runtime is usually an outlier
        mini_batch_times = mini_batch_times[1:]
        mini_batch_time = sum(mini_batch_times) / len(mini_batch_times)
        assert (0.75 * expected_mini_batch_times[cluster]
                < mini_batch_time
                < 1.25 * expected_mini_batch_times[cluster]), \
                'average mini-batch time is outside expected range'

    # Return test function from factory function
    func.__name__ = test_name
    return func

# Create test functions that can interact with PyTest
for _test_func in tools.create_tests(setup_experiment,
                                     __file__,
                                     nodes=num_nodes):
    globals()[_test_func.__name__] = augment_test_func(_test_func)
