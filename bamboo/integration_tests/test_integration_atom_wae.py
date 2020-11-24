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

# ATOM application directory
root_dir = os.path.dirname(os.path.dirname(current_dir))
atom_dir = os.path.join(root_dir, 'applications', 'ATOM')
sys.path.append(atom_dir)

# ==============================================
# Options
# ==============================================

# Training options
num_epochs = 10
mini_batch_size = 512
num_nodes = 2

# Reconstruction loss
expected_train_recon_range = (0.3, 0.4)
expected_test_recon_range = (0.3, 0.4)

# Average mini-batch time (in sec) for each LC system
# Note that run times are with LBANN_DETERMINISTIC set
# Commented out times are prior to thread safe RNGs
expected_mini_batch_times = {
    'lassen':   0.035,
}

#@todo: add other cluster if need be
vocab_loc = {
    'lassen': '/p/gpfs1/brainusr/datasets/atom/mpro_inhib/enamine_all2018q1_2020q1-2_mpro_inhib_kekulized.vocab'
}
# ==============================================
# Setup LBANN experiment
# ==============================================
def list2str(l):
    return ' '.join(l)

def setup_experiment(lbann):
    """Construct LBANN experiment.

    Args:
        lbann (module): Module for LBANN Python frontend

    """
    if tools.system(lbann) != 'lassen':
      message = f'{os.path.basename(__file__)} is only supported on lassen system'
      print('Skip - ' + message)
      pytest.skip(message)

    trainer = lbann.Trainer(mini_batch_size=mini_batch_size)
    model = construct_model(lbann)

    import data.atom
    data_reader = data.atom.make_data_reader(lbann)

    opt = lbann.Adam(learn_rate=3e-4, beta1=0.9, beta2=0.99, eps=1e-8)
    return trainer, model, data_reader, opt

def construct_model(lbann):
    """Construct LBANN model.
    Args:
        lbann (module): Module for LBANN Python frontend

    """

    import models.atom.wae as molwae

    pad_index = 40

    sequence_length = 100

    data_layout = "data_parallel"
    # Layer graph
    input_ = lbann.Identity(lbann.Input(name='inp',target_mode="N/A"), name='inp1')
    wae_loss= []
    input_feature_dims = sequence_length

    embedding_size = 42
    dictionary_size = 42

    z = lbann.Gaussian(mean=0.0,stdev=1.0, neuron_dims="128")
    recon, d1_real, d1_fake, d_adv, _ = molwae.MolWAE(input_feature_dims,
                                                      dictionary_size,
                                                      embedding_size,
                                                      pad_index)(input_,z)

    zero  = lbann.Constant(value=0.0,num_neurons='1',name='zero')
    one  = lbann.Constant(value=1.0,num_neurons='1',name='one')

    d1_real_bce = lbann.SigmoidBinaryCrossEntropy([d1_real,one],name='d1_real_bce')
    d1_fake_bce = lbann.SigmoidBinaryCrossEntropy([d1_fake,zero],name='d1_fake_bce')
    d_adv_bce = lbann.SigmoidBinaryCrossEntropy([d_adv,one],name='d_adv_bce')

    wae_loss.append(recon)

    layers = list(lbann.traverse_layer_graph(input_))
    # Setup objective function
    weights = set()
    src_layers = []
    dst_layers = []
    for l in layers:
      if(l.weights and "disc0" in l.name and "instance1" in l.name):
        src_layers.append(l.name)
      #freeze weights in disc2
      if(l.weights and "disc1" in l.name):
        dst_layers.append(l.name)
        for idx in range(len(l.weights)):
          l.weights[idx].optimizer = lbann.NoOptimizer()
      weights.update(l.weights)
    l2_reg = lbann.L2WeightRegularization(weights=weights, scale=1e-4)

    wae_loss.append(d1_real_bce)
    wae_loss.append(d_adv_bce)
    wae_loss.append(d1_fake_bce)
    wae_loss.append(l2_reg)

    obj = lbann.ObjectiveFunction(wae_loss)

    # Initialize check metric callback
    metrics = [lbann.Metric(recon, name='recon')]

    callbacks = [lbann.CallbackPrint(),
                 lbann.CallbackTimer()]

    callbacks.append(lbann.CallbackReplaceWeights(source_layers=list2str(src_layers),
                                 destination_layers=list2str(dst_layers),
                                 batch_interval=2))

    return lbann.Model(num_epochs,
                       weights=weights,
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
    def func(cluster, exes, dirname,weekly):

        if not weekly:
            pytest.skip('This app runs {} with weekly builds only'.format(test_name))

        # Run LBANN experiment
        experiment_output = test_func(cluster, exes, dirname)

        # Parse LBANN log file
        train_recon = None
        test_recon = None
        mini_batch_times = []
        with open(experiment_output['stdout_log_file']) as f:
            for line in f:
                match = re.search('training epoch [0-9]+ recon : ([0-9.]+)', line)
                if match:
                    train_recon = float(match.group(1))
                match = re.search('test recon : ([0-9.]+)', line)
                if match:
                    test_recon = float(match.group(1))
                match = re.search('training epoch [0-9]+ mini-batch time statistics : ([0-9.]+)s mean', line)
                if match:
                    mini_batch_times.append(float(match.group(1)))

        # Check if training reconstruction is within expected range
        assert (expected_train_recon_range[0]
                < train_recon
                < expected_train_recon_range[1]), \
                'train reconstruction loss is outside expected range'

        # Check if testing reconstruction  is within expected range
        assert (expected_test_recon_range[0]
                < test_recon
                < expected_test_recon_range[1]), \
                'test reconstruction loss is outside expected range'

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

m_lbann_args=f"--vocab={vocab_loc['lassen']} --sequence_length=100  --delimiter=0 "
# Create test functions that can interact with PyTest
for _test_func in tools.create_tests(setup_experiment,
                                     __file__,
                                     lbann_args=[m_lbann_args],
                                     nodes=num_nodes):
    globals()[_test_func.__name__] = augment_test_func(_test_func)
