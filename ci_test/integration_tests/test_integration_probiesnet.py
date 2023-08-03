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
lbann_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
app_path = os.path.join(lbann_dir, 'applications', 'physics','HRRL')
sys.path.append(app_path)
import tools

# ==============================================
# Options
# ==============================================

# Average mini-batch time (in sec) for each LC system
# Note that run times are with LBANN_DETERMINISTIC set
# Commented out times are prior to thread safe RNGs

################################################################################
# Weekly training options and targets
################################################################################
weekly_options_and_targets = {
    'num_nodes': 1,
    'num_epochs': 10,
    'mini_batch_size': 32,
    'expected_train_pc_range': (0.89, 0.92),
    'expected_test_pc_range': (0.90, 0.931),
    'fraction_of_data_to_use': 1.0,
    'expected_mini_batch_times': {
        'lassen':   0.0069, # Old as of 3/21/2022 0.0051,
        'pascal':   0.2267, # Old as of 3/21/2022 0.0146,
    }
}

################################################################################
# Nightly training options and targets
################################################################################
nightly_options_and_targets = {
    'num_nodes': 1,
    'num_epochs': 10,
    'mini_batch_size': 32,
    'expected_train_pc_range': (0.57, 0.601), # BVE changed from 0.60 on 3/9/23 - BVE changed from 0.59 on 9/21/22
    'expected_test_pc_range': (0.66, 0.68),
    'fraction_of_data_to_use': 0.01,
    'expected_mini_batch_times': {
        'lassen':   0.0069,
        'pascal':   0.0386, # BVE changed from 0.172, on 9/21/22
        'tioga':    0.0386, # BVE dummy value from pascal
        'corona':   0.0386, # BVE dummy value from pascal
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
    protobuf_file = os.path.join(app_path,'data',
                                 'probies_v2.prototext')

    message = lbann.lbann_pb2.LbannPB()
    with open(protobuf_file, 'r') as f:
        google.protobuf.text_format.Merge(f.read(), message)
    message = message.data_reader

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

    trainer = lbann.Trainer(mini_batch_size=options['mini_batch_size'])
    model = construct_model(lbann, options['num_epochs'])

    data_reader = make_data_reader(lbann, options['fraction_of_data_to_use'])

    opt = lbann.Adam(learn_rate=0.0002,beta1=0.9,beta2=0.99,eps=1e-8)
    return trainer, model, data_reader, opt, options['num_nodes']

def construct_model(lbann, num_epochs):
    """Construct LBANN model.
    Args:
        lbann (module): Module for LBANN Python frontend

    """

    import models.probiesNet as model

    images = lbann.Input(data_field='samples')
    responses = lbann.Input(data_field='responses')

    num_labels = 5

    images = lbann.Reshape(images, dims=[1, 300, 300])


    pred = model.PROBIESNet(num_labels)(images)

    mse = lbann.MeanSquaredError([responses, pred])

    # Pearson Correlation
    # rho(x,y) = covariance(x,y) / sqrt( variance(x) * variance(y) )
    pearson_r_cov = lbann.Covariance([pred, responses],
				   name="pearson_r_cov")

    pearson_r_var1 = lbann.Variance(responses,
				 name="pearson_r_var1")

    pearson_r_var2 = lbann.Variance(pred,
				name="pearson_r_var2")


    pearson_r_mult = lbann.Multiply([pearson_r_var1, pearson_r_var2],
				    name="pearson_r_mult")

    pearson_r_sqrt = lbann.Sqrt(pearson_r_mult,
		            name="pearson_r_sqrt")

    eps = lbann.Constant(value=1e-07,hint_layer=pearson_r_sqrt)
    pearson_r = lbann.Divide([pearson_r_cov, lbann.Add(pearson_r_sqrt,eps)],
			     name="pearson_r")


    metrics = [lbann.Metric(mse, name='mse')]
    metrics.append(lbann.Metric(pearson_r, name='pearson_r'))

    callbacks = [lbann.CallbackPrint(),
                 lbann.CallbackTimer()]


    layers = list(lbann.traverse_layer_graph([images, responses]))
    return lbann.Model(num_epochs,
                    layers=layers,
                    metrics=metrics,
                    objective_function=mse,
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
        train_pc = None
        test_pc = None
        mini_batch_times = []
        with open(experiment_output['stdout_log_file']) as f:
            for line in f:
                match = re.search('training epoch [0-9]+ pearson_r : ([0-9.]+)', line)
                if match:
                    train_pc = float(match.group(1))
                match = re.search('test pearson_r : ([0-9.]+)', line)
                if match:
                    test_pc = float(match.group(1))
                match = re.search('training epoch [0-9]+ mini-batch time statistics : ([0-9.]+)s mean', line)
                if match:
                    mini_batch_times.append(float(match.group(1)))

        # Check if training reconstruction is within expected range
        assert ((train_pc > targets['expected_train_pc_range'][0]
                 and train_pc < targets['expected_train_pc_range'][1])), \
                f"train pearson correlation {train_pc:.3f} is outside expected range " + \
                f"[{targets['expected_train_pc_range'][0]:.3f},{targets['expected_train_pc_range'][1]:.3f}]"

        # Check if testing reconstruction  is within expected range
        assert ((test_pc > targets['expected_test_pc_range'][0]
                 and test_pc < targets['expected_test_pc_range'][1])), \
                f"test pearson correlation {test_pc:.3f} is outside expected range " + \
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

m_lbann_args=f"--use_data_store --preload_data_store"
# Create test functions that can interact with PyTest
for _test_func in tools.create_tests(setup_experiment,
                                     __file__,
                                     lbann_args=[m_lbann_args]):
    globals()[_test_func.__name__] = augment_test_func(_test_func)
