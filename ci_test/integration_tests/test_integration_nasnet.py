import functools
import operator
import os
import os.path
import re
import sys
import numpy as np
import google.protobuf.text_format
import pytest
from os.path import abspath, dirname, join, realpath
import tools
import warnings
import shutil

# Local files
current_file = os.path.realpath(__file__)
current_dir = os.path.dirname(current_file)
sys.path.insert(0, os.path.join(os.path.dirname(current_dir), 'common_python'))
lbann_dir = dirname(os.path.dirname(os.path.dirname(current_file)))
app_path = join(lbann_dir, 'applications', 'nas', 'nasnet')
sys.path.append(app_path)
import cifar_networks

#module_file = os.path.abspath(__file__)
module_name = os.path.splitext(os.path.basename(current_file))[0]
#module_dir = os.path.dirname(module_file)
expd = os.path.join(current_dir,'experiments',module_name)
print('Current file, expd  ', current_file, expd)
# ==============================================
# Options
# ==============================================

# Training options
proc_per_trainer=1
proc_per_node=2

# Average mini-batch time (in sec) for each LC system
# Note that run times are with LBANN_DETERMINISTIC set
# Commented out times are prior to thread safe RNGs

################################################################################
# Weekly training options and targets
################################################################################
weekly_options_and_targets = {
    'num_nodes': 2,
    'num_epochs': 10,
    'mini_batch_size': 64,
    'expected_train_accuracy_range': (70, 85),
    'expected_test_accuracy_range': (70, 85),
    'expected_mini_batch_times': {
        'lassen':   0.075,
        'pascal':   0.15,
    }
}

################################################################################
# Nightly training options and targets
################################################################################
nightly_options_and_targets = {
    'num_nodes': 2,
    'num_epochs': 4,
    'mini_batch_size': 64,
    'expected_train_accuracy_range': (47.9, 65), # BVE relaxed lower bound from 50 9/21/22, TBN relaxed further to 47.9 5/16/23
    'expected_test_accuracy_range': (49, 65), # BVE relaxed lower bound from 50 9/22/22
    'expected_mini_batch_times': {
        'lassen':   0.075,
        'pascal':   0.18, # BVE increased from 0.15, 9/21/22
        'tioga':    0.18, # BVE dummy value from pascal
        'corona':   0.18, # BVE dummy value from pascal
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
    if tools.system(lbann) != 'lassen' and tools.system(lbann) != 'pascal' :
        message = f'{os.path.basename(__file__)} is only supported on lassen, and pascal systems'
        print('Skip - ' + message)
        pytest.skip(message)

    if weekly:
        options = weekly_options_and_targets
    else:
        options = nightly_options_and_targets

    test_fname = 'test_integration_nasnet_v1'

    mini_batch_size=options['mini_batch_size']
    num_epochs = options['num_epochs']
    if tools.system(lbann) == 'lassen':
        proc_per_node = 4
    else:
        proc_per_node = 2

    pop_size = int(options['num_nodes']*proc_per_node/proc_per_trainer)
    trainer, model, reader, opt =  cifar_networks.create_networks(expd,
                                                                  num_epochs,
                                                                  mini_batch_size,
                                                                  pop_size)

    return trainer, model, reader, opt, options['num_nodes']



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
        num_trainers = None
        train_accuracy = None
        test_accuracy = None
        mini_batch_times = []
        with open(experiment_output['stdout_log_file']) as f:
            for line in f:
                if num_trainers is None:
                    match = re.search('Trainers *: ([0-9]+)', line)
                    if match:
                        num_trainers = int(match.group(1))

                match = re.search('training epoch [0-9]+ accuracy : ([0-9.]+)', line)
                if match:
                    train_accuracy = float(match.group(1))
                match = re.search('test accuracy : ([0-9.]+)', line)
                if match:
                    test_accuracy = float(match.group(1))
                match = re.search('training epoch [0-9]+ mini-batch time statistics : ([0-9.]+)s mean', line)
                if match:
                    mini_batch_times.append(float(match.group(1)))

        #Check that all trainers complete training and evaluation
        assert (len(mini_batch_times) % num_trainers == 0), \
                'output log is missing values and metrics from one or more trainers'

        # Check if training accuracy is within expected range
        assert ((train_accuracy > targets['expected_train_accuracy_range'][0]
                 and train_accuracy < targets['expected_train_accuracy_range'][1])), \
                f"train accuracy {train_accuracy:.3f} is outside expected range " + \
                f"[{targets['expected_train_accuracy_range'][0]:.3f},{targets['expected_train_accuracy_range'][1]:.3f}]"

        # Check if testing accuracy  is within expected range
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


if os.path.isdir(expd):
   shutil.rmtree(expd)

os.makedirs(expd)
#proto_file_name="experiment.prototext.trainer"+str(network_id),
proto_file = os.path.join(expd,'experiment.prototext.trainer0')
#m_lbann_args=f"--procs_per_trainer={proc_per_trainer} --generate_multi_proto --proto_file_name={proto_file}"
m_lbann_args=f"--procs_per_trainer={proc_per_trainer} --generate_multi_proto"
# Create test functions that can interact with PyTest
for _test_func in tools.create_tests(setup_experiment,
                                     __file__,
                                     proto_file_name=proto_file,
                                     time_limit=10,
                                     lbann_args=[m_lbann_args]):
    globals()[_test_func.__name__] = augment_test_func(_test_func)
