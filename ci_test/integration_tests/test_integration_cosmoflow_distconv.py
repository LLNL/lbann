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
import lbann.contrib.args
from lbann.core.util import get_parallel_strategy_args

# Local files
current_file = realpath(__file__)
lbann_dir = dirname(os.path.dirname(os.path.dirname(current_file)))
app_path = join(lbann_dir, 'applications', 'physics', 'cosmology', 'cosmoflow')
sys.path.append(app_path)

# ==============================================
# Options
# ==============================================

# Training options
procs_per_node = 2#2 # Only use 2 GPUs to ensure comparable testing between lassen and pascal
                   # this model is very sensitive to differences in how it is initialized
                   # and parallelized

#model_zoo_dir = dirname(app_path)
# data_reader_prototext = join(model_zoo_dir,
#                              'data',
#                              'jag_conduit_reader.prototext')
# metadata_prototext = join(model_zoo_dir,
#                              'data',
#                              'jag_100M_metadata.prototext')

# ydim = 16399 # image+scalar dim (default: 64*64*4+15=16399)
# zdim = 20 # latent space dim (default: 20)
# mcf = 1 # model capacity factor (default: 1)
# useCNN = False

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
    'num_nodes': 2,
    'num_epochs': 10,
    'mini_batch_size': 2,
    'input_width': 128,
    'num_secrets': 4,
    'use_batchnorm': True,
    'local_batchnorm': True,
    'depth_groups': 4, #2,
    'sample_groups': 1,
    'learning_rate': 0.001,
#    'min_distconv_width': 4,
    'mlperf': True,
    'transform_input': False, #True,
    'expected_train_mse_range': (0.56, 1.79), #(0.273, 0.290),
    'expected_test_mse_range':  (1.15, 1.79), #(0.118, 0.120),
#    'expected_test_mse_range': (2.96, 2.97),
    'fraction_of_data_to_use': 1.0,
    'expected_mini_batch_times': {
        'lassen':   0.035, #0.0229,
        'pascal':   0.044,
        'tioga':   0.069, #0.044,
        'corona':   0.14,
    }
}

# ==============================================
# Setup LBANN experiment
# ==============================================

def make_data_reader(lbann, fraction_of_data_to_use):
    """Create a data reader for CosmoFlow.

    Args:
        {train, val, test}_path (str): Path to the corresponding dataset.
        num_responses (int): The number of parameters to predict.
    """

#python3 train_cosmoflow.py --train-dir= --val-dir= --test-dir= --nodes=1 --procs-per-node=2 --depth-groups=2

    reader_args = [
        {"role": "train", "data_filename": "/p/vast1/lbann/datasets/cosmoflow/cosmoUniverse_2019_05_4parE_tf_v2_hdf5_tiny/train"},
        {"role": "validate", "data_filename": "/p/vast1/lbann/datasets/cosmoflow/cosmoUniverse_2019_05_4parE_tf_v2_hdf5_tiny/validation"},
        {"role": "test", "data_filename": "/p/vast1/lbann/datasets/cosmoflow/cosmoUniverse_2019_05_4parE_tf_v2_hdf5_tiny/validation"},
    ]

    for reader_arg in reader_args:
        reader_arg["data_file_pattern"] = "{}/*.hdf5".format(
            reader_arg["data_filename"])
        reader_arg["hdf5_key_data"] = "full"
        reader_arg["hdf5_key_responses"] = "unitPar"
        reader_arg["num_responses"] = 4 #options['num_secrets']
        reader_arg.pop("data_filename")

    readers = []
    for reader_arg in reader_args:
        reader = lbann.reader_pb2.Reader(
            name="hdf5",
            shuffle=(reader_arg["role"] != "test"),
            validation_fraction=0,
            absolute_sample_count=0,
            fraction_of_data_to_use=1.0,
            disable_labels=True,
            disable_responses=False,
            scaling_factor_int16=1.0,
            **reader_arg)

        readers.append(reader)

    return lbann.reader_pb2.DataReader(reader=readers)

def setup_experiment(lbann, weekly):
    """Construct LBANN experiment.

    Args:
        lbann (module): Module for LBANN Python frontend

    """
    if not lbann.has_feature('DISTCONV'):
      message = f'{os.path.basename(__file__)} requires DistConv support'
      print('Skip - ' + message)
      pytest.skip(message)

    # FIXME: Remove this check after Pack/Unpack PR on H2 merges.
    # if tools.system(lbann) != 'lassen' and tools.system(lbann) != 'pascal' and tools.system(lbann) != 'tioga':
    #   message = f'{os.path.basename(__file__)} is only supported on lassen, tioga, and pascal systems'
    #   print('Skip - ' + message)
#      pytest.skip(message)

    if weekly:
        options = weekly_options_and_targets
    else:
        options = nightly_options_and_targets

    trainer = lbann.Trainer(mini_batch_size=options['mini_batch_size'],
                            serialize_io=True)

    # Checkpoint after every epoch
    # trainer.callbacks = [
    #     lbann.CallbackCheckpoint(
    #         checkpoint_dir='ckpt',
    #         checkpoint_epochs=1,
    #         checkpoint_steps=1
    #     )
    # ]

    # Set parallel_strategy
    parallel_strategy = get_parallel_strategy_args(
        sample_groups=options['sample_groups'],
        depth_groups=options['depth_groups'])
    import cosmoflow_model
    model = cosmoflow_model.construct_cosmoflow_model(parallel_strategy=parallel_strategy,
                                                      local_batchnorm=options['local_batchnorm'],
                                                      input_width=options['input_width'],
                                                      num_secrets=options['num_secrets'],
                                                      use_batchnorm=options['use_batchnorm'],
                                                      num_epochs=options['num_epochs'],
                                                      learning_rate=options['learning_rate'],
                                                      min_distconv_width=options['depth_groups'],
                                                      mlperf=options['mlperf'],
                                                      transform_input=options['transform_input'])

    # model.callbacks.append(lbann.CallbackDebug())
    # Setup optimizer
    opt = lbann.Adam(learn_rate=0.001,beta1=0.9,beta2=0.99,eps=1e-8)
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
        train_mse = None
        test_mse = None
        mini_batch_times = []
        with open(experiment_output['stdout_log_file']) as f:
            for line in f:
                match = re.search('training epoch [0-9]+ MSE : ([0-9.]+)', line)
                if match:
                    train_mse = float(match.group(1))
                match = re.search('test MSE : ([0-9.]+)', line)
                if match:
                    test_mse = float(match.group(1))
                match = re.search('training epoch [0-9]+ mini-batch time statistics : ([0-9.]+)s mean', line)
                if match:
                    mini_batch_times.append(float(match.group(1)))

        # Check if training mean squared error is within expected range
        assert (targets['expected_train_mse_range'][0]
                < train_mse
                < targets['expected_train_mse_range'][1]), \
                'train mean squared error error is outside expected range'

        # Check if testing mean squared error  is within expected range
        assert (targets['expected_test_mse_range'][0]
                < test_mse
                < targets['expected_test_mse_range'][1]), \
                'test mean squared error error is outside expected range'

        # Check if mini-batch time is within expected range
        # Note: Skip first epoch since its runtime is usually an outlier
        mini_batch_times = mini_batch_times[1:]
        mini_batch_time = sum(mini_batch_times) / len(mini_batch_times)
        assert (0.75 * targets['expected_mini_batch_times'][cluster]
                < mini_batch_time
                < 1.25 * targets['expected_mini_batch_times'][cluster]), \
                'average mini-batch time is outside expected range'

    # Return test function from factory function
    func.__name__ = test_name
    return func

m_lbann_args=f"--use_data_store"
m_environment = lbann.contrib.args.get_distconv_environment(
    num_io_partitions=nightly_options_and_targets['depth_groups'])
m_environment['LBANN_KEEP_ERROR_SIGNALS'] = 1
m_environment['SPDLOG_LEVEL'] = "error"
#m_environment['LBANN_DISTCONV_DETERMINISTIC'] = 0
# Create test functions that can interact with PyTest
for _test_func in tools.create_tests(setup_experiment,
                                     __file__,
                                     time_limit=10,
                                     lbann_args=[m_lbann_args],
                                     environment = m_environment,
                                     procs_per_node=procs_per_node):
    globals()[_test_func.__name__] = augment_test_func(_test_func)
