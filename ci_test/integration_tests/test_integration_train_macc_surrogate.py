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

# Local files
current_file = realpath(__file__)
lbann_dir = dirname(os.path.dirname(os.path.dirname(current_file)))
app_path = join(lbann_dir, 'applications', 'physics','ICF')
sys.path.append(app_path)

# ==============================================
# Options
# ==============================================

# Training options
num_epochs = 10
mini_batch_size = 128
num_nodes = 1
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

xdim = 5 # input (x) dim (default: 5)
ydim = 16399 # image+scalar dim (default: 64*64*4+15=16399)
zdim = 20 # latent space dim (default: 20)
wae_mcf = 1 # model capacity factor (default: 1)
surrogate_mcf = 1 # model capacity factor (default: 1)
lambda_cyc = 1e-3 # lambda-cyc (default: 1e-3)

useCNN = False

# Reconstruction loss
expected_train_range = (0.42, 0.44)
expected_test_range = (0.48, 0.50)

# Average mini-batch time (in sec) for each LC system
# Note that run times are with LBANN_DETERMINISTIC set
# Commented out times are prior to thread safe RNGs
expected_mini_batch_times = {
    'lassen':   0.0530066,
    'pascal':   0.044,
}
# ==============================================
# Setup LBANN experiment
# ==============================================

def make_data_reader(lbann):
    """Make Protobuf message for HRRL  data reader.

    """
    import lbann.contrib.lc.paths

    # Load data readers from prototext
    message = lbann.lbann_pb2.LbannPB()
    with open(data_reader_prototext, 'r') as f:
        google.protobuf.text_format.Merge(f.read(), message)
    message = message.data_reader

    # Use less training data for the integration test
    message.reader[0].percent_of_data_to_use = 0.01

    # Set paths
    return message

def setup_experiment(lbann):
    """Construct LBANN experiment.

    Args:
        lbann (module): Module for LBANN Python frontend

    """
    if tools.system(lbann) != 'lassen' and tools.system(lbann) != 'pascal':
      message = f'{os.path.basename(__file__)} is only supported on lassen and pascal systems'
      print('Skip - ' + message)
      pytest.skip(message)

    trainer = lbann.Trainer(mini_batch_size=mini_batch_size,
                            serialize_io=True)
    import macc_models
    dump_models = 'dump_models'
    ltfb_batch_interval = 0
    pretrained_dir = ' '
    model = macc_models.construct_macc_surrogate_model(xdim=xdim,
                                                       ydim=ydim,
                                                       zdim=zdim,
                                                       wae_mcf=wae_mcf,
                                                       surrogate_mcf=surrogate_mcf,
                                                       lambda_cyc=lambda_cyc,
                                                       useCNN=useCNN,
                                                       dump_models=dump_models,
                                                       pretrained_dir=pretrained_dir,
                                                       ltfb_batch_interval=ltfb_batch_interval,
                                                       num_epochs=num_epochs)

    # Setup optimizer
    opt = lbann.Adam(learn_rate=0.0001,beta1=0.9,beta2=0.99,eps=1e-8)
    # Load data reader from prototext
    data_reader = make_data_reader(lbann)

    return trainer, model, data_reader, opt

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
        train_pc = None
        test_pc = None
        mini_batch_times = []
        with open(experiment_output['stdout_log_file']) as f:
            for line in f:
                match = re.search('training epoch [0-9]+ output cycle loss : ([0-9.]+)', line)
                if match:
                    train_pc = float(match.group(1))
                match = re.search('test output cycle loss : ([0-9.]+)', line)
                if match:
                    test_pc = float(match.group(1))
                match = re.search('training epoch [0-9]+ mini-batch time statistics : ([0-9.]+)s mean', line)
                if match:
                    mini_batch_times.append(float(match.group(1)))

        # Check if training reconstruction is within expected range
        assert (expected_train_range[0]
                < train_pc
                < expected_train_range[1]), \
                'train reconstruction error is outside expected range'

        # Check if testing reconstruction  is within expected range
        assert (expected_test_range[0]
                < test_pc
                < expected_test_range[1]), \
                'test reconstruction error is outside expected range'

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

m_lbann_args=f"--use_data_store --preload_data_store --metadata={metadata_prototext}"
# Create test functions that can interact with PyTest
for _test_func in tools.create_tests(setup_experiment,
                                     __file__,
                                     lbann_args=[m_lbann_args],
                                     procs_per_node=procs_per_node,
                                     nodes=num_nodes):
    globals()[_test_func.__name__] = augment_test_func(_test_func)
