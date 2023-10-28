import os
import os.path
import re
import sys
import pytest
from os.path import dirname, join, realpath
import warnings
import tools

# Local files
current_file = realpath(__file__)
lbann_dir = dirname(os.path.dirname(os.path.dirname(current_file)))
app_path = join(lbann_dir, 'applications', 'gan','mnist')
sys.path.append(app_path)

# ==============================================
# Options
# ==============================================

# Training options
procs_per_node = 1 # Only use 1 GPU

################################################################################
# Weekly training options and targets
################################################################################
weekly_options_and_targets = {
    'num_nodes': 1,
    'num_epochs': 10,
    'mini_batch_size': 128,
    'expected_train_range': (1.02, 1.04),
    'fraction_of_data_to_use': 1,
    'expected_mini_batch_times': {
        'lassen':   0.005,
        'pascal':   0.005,
    }
}

################################################################################
# Nightly training options and targets
################################################################################
nightly_options_and_targets = {
    'num_nodes': 1,
    'num_epochs': 10,
    'mini_batch_size': 128,
    'expected_train_range': (0.509, 0.53), # BVE Relexed the range for 0.51 10-28-2023
    'fraction_of_data_to_use': 0.1,
    'expected_mini_batch_times': {
        'lassen':   0.005,
        'pascal':   0.005,
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
    if tools.system(lbann) != 'lassen' and tools.system(lbann) != 'pascal':
      message = f'{os.path.basename(__file__)} is only supported on lassen and pascal systems'
      print('Skip - ' + message)
      pytest.skip(message)

    if weekly:
        options = weekly_options_and_targets
    else:
        options = nightly_options_and_targets

    trainer = lbann.Trainer(mini_batch_size=options['mini_batch_size'],
                            serialize_io=True)
    import gan_model
    model = gan_model.build_model(num_epochs=options['num_epochs'])

    # Setup optimizer
    opt = lbann.Adam(learn_rate=1e-4, beta1=0., beta2=0.99, eps=1e-8)
    # Load data reader from prototext
    from mnist_dataset import make_data_reader
    data_reader = make_data_reader(options['fraction_of_data_to_use'])

    return trainer, model, data_reader, opt, options['num_nodes']

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
        train_pc = None
        mini_batch_times = []
        with open(experiment_output['stdout_log_file']) as f:
            for line in f:
                match = re.search('training epoch [0-9]+ objective function : ([0-9.]+)', line)
                if match:
                    train_pc = float(match.group(1))
                match = re.search('training epoch [0-9]+ mini-batch time statistics : ([0-9.]+)s mean', line)
                if match:
                    mini_batch_times.append(float(match.group(1)))

        # Check if training reconstruction is within expected range
        assert ((train_pc > targets['expected_train_range'][0]
                 and train_pc < targets['expected_train_range'][1])), \
                f"train objective function {train_pc:.3f} is outside expected range " + \
                f"[{targets['expected_train_range'][0]:.3f},{targets['expected_train_range'][1]:.3f}]"

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
                                     procs_per_node=procs_per_node):
    globals()[_test_func.__name__] = augment_test_func(_test_func)
