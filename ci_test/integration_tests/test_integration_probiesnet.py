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
lbann_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
app_path = os.path.join(lbann_dir, 'applications', 'physics','HRRL')
sys.path.append(app_path)
import tools

# ==============================================
# Options
# ==============================================

# Training options
num_epochs = 10
mini_batch_size = 32
num_nodes = 1

# Reconstruction loss
expected_train_pc_range = (0.89, 0.92)
expected_test_pc_range = (0.90, 0.925)

# Average mini-batch time (in sec) for each LC system
# Note that run times are with LBANN_DETERMINISTIC set
# Commented out times are prior to thread safe RNGs
expected_mini_batch_times = {
    'lassen':   0.0051,
    'pascal':   0.0146,
}
# ==============================================
# Setup LBANN experiment
# ==============================================
def list2str(l):
    return ' '.join(l)

def make_data_reader(lbann):
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

    trainer = lbann.Trainer(mini_batch_size=mini_batch_size)
    model = construct_model(lbann)

    data_reader = make_data_reader(lbann)

    opt = lbann.Adam(learn_rate=0.0002,beta1=0.9,beta2=0.99,eps=1e-8)
    return trainer, model, data_reader, opt

def construct_model(lbann):
    """Construct LBANN model.
    Args:
        lbann (module): Module for LBANN Python frontend

    """

    import models.probiesNet as model

    images = lbann.Input(data_field='samples')
    responses = lbann.Input(data_field='responses')

    num_labels = 5

    images = lbann.Reshape(images, dims='1 300 300')


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
    def func(cluster, dirname,weekly):

        if not weekly:
            pytest.skip('This app runs {} with weekly builds only'.format(test_name))

        # Run LBANN experiment
        experiment_output = test_func(cluster, dirname)

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
        assert (expected_train_pc_range[0]
                < train_pc
                < expected_train_pc_range[1]), \
                'train pearson correlation is outside expected range'

        # Check if testing reconstruction  is within expected range
        assert (expected_test_pc_range[0]
                < test_pc
                < expected_test_pc_range[1]), \
                'test pearson correlation is outside expected range'

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

m_lbann_args=f"--use_data_store --preload_data_store"
# Create test functions that can interact with PyTest
for _test_func in tools.create_tests(setup_experiment,
                                     __file__,
                                     lbann_args=[m_lbann_args],
                                     nodes=num_nodes):
    globals()[_test_func.__name__] = augment_test_func(_test_func)
