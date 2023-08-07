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
import warnings
import tools

# Local files
current_file = realpath(__file__)
lbann_dir = dirname(os.path.dirname(os.path.dirname(current_file)))
app_path = join(lbann_dir, 'applications', 'physics','ICF')
sys.path.append(app_path)

# ==============================================
# Options
# ==============================================

# Training options
procs_per_node = 2 # Only use 2 GPUs to ensure comparable testing between lassen and pascal
                   # this model is very sensitive to differences in how it is initialized
                   # and parallelized

model_zoo_dir = dirname(app_path)
data_reader_prototext = join(model_zoo_dir,
                             'data',
                             'jag_conduit_reader.prototext')
metadata_prototext = join(model_zoo_dir,
                             'data',
                             'jag_100M_metadata.prototext')

ydim = 16399 # image+scalar dim (default: 64*64*4+15=16399)
zdim = 20 # latent space dim (default: 20)
mcf = 1 # model capacity factor (default: 1)
useCNN = False

# Average mini-batch time (in sec) for each LC system
# Note that run times are with LBANN_DETERMINISTIC set
# Commented out times are prior to thread safe RNGs

################################################################################
# Weekly training options and targets
################################################################################
weekly_options_and_targets = {
    'num_nodes': 1,
    'num_epochs': 10,
    'mini_batch_size': 128,
    'expected_train_pc_range': (7.7, 7.9),
    'expected_test_pc_range': (8.0, 8.2),
    'fraction_of_data_to_use': 0.1,
    'expected_mini_batch_times': {
        'lassen':   0.0530066,
        'pascal':   0.044,
    }
}

################################################################################
# Nightly training options and targets
################################################################################
nightly_options_and_targets = {
    'num_nodes': 1,
    'num_epochs': 10,
    'mini_batch_size': 128,
    'expected_train_pc_range': (19.9, 20.1), # BVE Changing the limits from 20.0
    'expected_test_pc_range': (19.1, 19.2),
    'fraction_of_data_to_use': 0.01,
    'expected_mini_batch_times': {
        'lassen':   0.0530066,
        'pascal':   0.044,
    }
}

# ==============================================
# Setup LBANN experiment
# ==============================================

def make_data_reader(lbann, fraction_of_data_to_use):
    """Make Protobuf message for HRRL  data reader.

    """
    import lbann.contrib.lc.paths

    # Load data readers from prototext
    message = lbann.lbann_pb2.LbannPB()
    with open(data_reader_prototext, 'r') as f:
        google.protobuf.text_format.Merge(f.read(), message)
    message = message.data_reader

    # Use less training data for the integration test
    message.reader[0].fraction_of_data_to_use = fraction_of_data_to_use

    # Set paths
    return message

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
    import macc_models
    dump_models = 'dump_models'
    ltfb_batch_interval = 0
    model = macc_models.construct_jag_wae_model(ydim=ydim,
                                                zdim=zdim,
                                                mcf=mcf,
                                                useCNN=useCNN,
                                                dump_models=dump_models,
                                                ltfb_batch_interval=ltfb_batch_interval,
                                                num_epochs=options['num_epochs'])

    # Setup optimizer
    opt = lbann.Adam(learn_rate=0.0001,beta1=0.9,beta2=0.99,eps=1e-8)
    # Load data reader from prototext
    data_reader = make_data_reader(lbann, options['fraction_of_data_to_use'])

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
        test_pc = None
        mini_batch_times = []
        with open(experiment_output['stdout_log_file']) as f:
            for line in f:
                match = re.search('training epoch [0-9]+ recon_error : ([0-9.]+)', line)
                if match:
                    train_pc = float(match.group(1))
                match = re.search('test recon_error : ([0-9.]+)', line)
                if match:
                    test_pc = float(match.group(1))
                match = re.search('training epoch [0-9]+ mini-batch time statistics : ([0-9.]+)s mean', line)
                if match:
                    mini_batch_times.append(float(match.group(1)))

        # Check if training reconstruction is within expected range
        assert ((train_pc > targets['expected_train_pc_range'][0]
                 and train_pc < targets['expected_train_pc_range'][1])), \
                f"train reconstruction error {train_pc:.3f} is outside expected range " + \
                f"[{targets['expected_train_pc_range'][0]:.3f},{targets['expected_train_pc_range'][1]:.3f}]"

        # Check if testing reconstruction  is within expected range
        assert ((test_pc > targets['expected_test_pc_range'][0]
                 and test_pc < targets['expected_test_pc_range'][1])), \
                f"test reconstruction error {test_pc:.3f} is outside expected range " + \
                f"[{targets['expected_test_pc_range'][0]:.3f},{targets['expected_test_pc_range'][1]:.3f}]"

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

m_lbann_args=f"--use_data_store --preload_data_store --metadata={metadata_prototext}"
# Create test functions that can interact with PyTest
for _test_func in tools.create_tests(setup_experiment,
                                     __file__,
                                     lbann_args=[m_lbann_args],
                                     procs_per_node=procs_per_node):
    globals()[_test_func.__name__] = augment_test_func(_test_func)
