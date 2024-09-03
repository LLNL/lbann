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
import data.imagenet

# ==============================================
# Options
# ==============================================

# Training options
imagenet_fraction = 0.280994  # Train with 360K out of 1.28M samples

# Top-5 classification accuracy (percent)

################################################################################
# Weekly training options and targets
################################################################################
# Reconstruction loss
weekly_options_and_targets = {
    'num_nodes': 4,
    'num_epochs': 5,
    'mini_batch_size': 256,
    'expected_train_accuracy_range': (45, 50),
    'expected_test_accuracy_range': (40, 55),
    'fraction_of_data_to_use': imagenet_fraction,
    'expected_mini_batch_times': {
        'pascal': 0.25,
        'lassen': 0.10,
        'ray':    0.15,
        'tioga':  0.25,
        'corona':  0.61,
    }
}

################################################################################
# Nightly training options and targets
################################################################################
nightly_options_and_targets = {
    'num_nodes': 2,
    'num_epochs': 3,
    'mini_batch_size': 256,
    'expected_train_accuracy_range': (2.75, 4.25), # Decreased lower limit from 3.0 to 2.75 due to variance
    'expected_test_accuracy_range': (1.5, 2.11), # BVE increased upper limit from 2.1 10/28
# 2.144 - Pascal
    # 1.446 - Corona
    'fraction_of_data_to_use': imagenet_fraction * 0.01,
    'expected_mini_batch_times': {
        'pascal': 0.43,
        'lassen': 0.15,
        'ray':    0.23,
        'tioga':  0.43,
        'corona':  0.61,
    }
}

# ==============================================
# Setup LBANN experiment
# ==============================================

def setup_experiment(lbann, weekly):
    """Construct LBANN experiment.

    Args:
        lbann (module): Module for LBANN Python frontend

    """
    if not lbann.has_feature('OPENCV') :
        message = f'{os.path.basename(__file__)} requires VISION support with OPENCV'
        print('Skip - ' + message)
        pytest.skip(message)

    # Skip test on CPU systems
    if not lbann.has_feature('GPU'):
        pytest.skip('only run {} on GPU systems'.format(test_name))

    if weekly:
        options = weekly_options_and_targets
    else:
        options = nightly_options_and_targets

    trainer = lbann.Trainer(mini_batch_size=options['mini_batch_size'])
    model = construct_model(lbann, options['num_epochs'])
    # Setup data reader
    data_reader = data.imagenet.make_data_reader(lbann, num_classes=1000)
    # We train on a subset of ImageNet
    data_reader.reader[0].fraction_of_data_to_use = options['fraction_of_data_to_use']
    # Only evaluate on ImageNet validation set at end of training
    data_reader.reader[1].role = 'test'

    optimizer = lbann.SGD(learn_rate=0.1, momentum=0.9)
    return trainer, model, data_reader, optimizer, options['num_nodes']

def construct_model(lbann, num_epochs):
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
    x = lbann.models.ResNet50(1000, bn_statistics_group_size=-1)(images)
    probs = lbann.Softmax(x)
    cross_entropy = lbann.CrossEntropy(probs, labels)
    top5 = lbann.TopKCategoricalAccuracy(probs, labels, k=5)
    layers = list(lbann.traverse_layer_graph(x))

    # Setup objective function
    l2_reg_weights = set()
    for l in layers:
        if type(l) == lbann.Convolution or type(l) == lbann.FullyConnected:
            l2_reg_weights.update(l.weights)
    l2_reg = lbann.L2WeightRegularization(weights=l2_reg_weights, scale=1e-4)
    obj = lbann.ObjectiveFunction([cross_entropy, l2_reg])

    # Objects for LBANN model
    callbacks = [lbann.CallbackPrint(), lbann.CallbackTimer()]
    metrics = [lbann.Metric(top5, name='top-5 accuracy', unit='%')]

    # Construct model
    return lbann.Model(num_epochs,
                       layers=layers,
                       objective_function=obj,
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

        if weekly:
            targets = weekly_options_and_targets
        else:
            targets = nightly_options_and_targets

        # Run LBANN experiment
        experiment_output = test_func(cluster, dirname, weekly)

        # Parse LBANN log file
        train_accuracy = None
        test_accuracy = None
        mini_batch_times = []
        with open(experiment_output['stdout_log_file']) as f:
            for line in f:
                match = re.search('training epoch [0-9]+ top-5 accuracy : ([0-9.]+)%', line)
                if match:
                    train_accuracy = float(match.group(1))
                match = re.search('test top-5 accuracy : ([0-9.]+)%', line)
                if match:
                    test_accuracy = float(match.group(1))
                match = re.search('training epoch [0-9]+ mini-batch time statistics : ([0-9.]+)s mean', line)
                if match:
                    mini_batch_times.append(float(match.group(1)))

        # Check if training accuracy is within expected range
        assert ((train_accuracy > targets['expected_train_accuracy_range'][0]
                 and train_accuracy < targets['expected_train_accuracy_range'][1])), \
                f"train accuracy {train_accuracy:.3f} is outside expected range " + \
                f"[{targets['expected_train_accuracy_range'][0]:.3f},{targets['expected_train_accuracy_range'][1]:.3f}]"

        # Check if testing accuracy is within expected range
        assert ((test_accuracy > targets['expected_test_accuracy_range'][0]
                 and test_accuracy < targets['expected_test_accuracy_range'][1])), \
                f"test accuracy {test_accuracy:.3f} is outside expected range " + \
                f"[{targets['expected_test_accuracy_range'][0]:.3f},{targets['expected_test_accuracy_range'][1]:.3f}]"

        # Check if mini-batch time is within expected range
        # Note: Skip first epoch since its runtime is usually an outlier
        mini_batch_times = mini_batch_times[1:]
        mini_batch_time = sum(mini_batch_times) / len(mini_batch_times)
        min_expected_mini_batch_time = 0.75 * targets['expected_mini_batch_times'][cluster]
        max_expected_mini_batch_time = 1.25 * targets['expected_mini_batch_times'][cluster]
        if (mini_batch_time < min_expected_mini_batch_time or
            mini_batch_time > max_expected_mini_batch_time):
            warnings.warn(f'average mini-batch time {mini_batch_time:.3f} is outside expected range ' +
                          f'[{min_expected_mini_batch_time:.3f}, {max_expected_mini_batch_time:.3f}]', UserWarning)

    # Return test function from factory function
    func.__name__ = test_name
    return func

# Create test functions that can interact with PyTest
for _test_func in tools.create_tests(setup_experiment,
                                     __file__,
                                     time_limit=30, # For the time being the bootstrap time for ROCm is slow
                                     lbann_args=['--load_full_sample_list_once']):
    globals()[_test_func.__name__] = augment_test_func(_test_func)
