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
current_file = os.path.realpath(__file__)
current_dir = os.path.dirname(current_file)
sys.path.insert(0, os.path.join(os.path.dirname(current_dir), 'common_python'))
lbann_dir = dirname(os.path.dirname(os.path.dirname(current_file)))
app_path = join(lbann_dir, 'applications', 'ATOM')
sys.path.append(app_path)

# ==============================================
# Options
# ==============================================

# Training options
num_epochs = 10
mini_batch_size = 512
num_nodes = 2
num_decoder_layers = 3

# Average mini-batch time (in sec) for each LC system
# Note that run times are with LBANN_DETERMINISTIC set
# Commented out times are prior to thread safe RNGs

################################################################################
# Weekly training options and targets
################################################################################
weekly_options_and_targets = {
    'num_nodes': 2,
    'num_epochs': 10,
    'mini_batch_size': 512,
    'num_decoder_layers': 3,
    'expected_train_recon_range': (0.95, 0.97),
    'expected_test_recon_range': (0.925, 0.94),
    'fraction_of_data_to_use': 0.01,
    'expected_mini_batch_times': {
        'lassen':   0.157,
        'pascal':   0.468, # BVE increase value from 0.365, 11/7/22
        'ray':   0.185,
    }
}

################################################################################
# Nightly training options and targets
################################################################################
nightly_options_and_targets = {
    'num_nodes': 2,
    'num_epochs': 10,
    'mini_batch_size': 512,
    'num_decoder_layers': 3,
    'expected_train_recon_range': (0.95, 0.97),
    'expected_test_recon_range': (0.925, 0.94),
    'fraction_of_data_to_use': 0.01,
    'expected_mini_batch_times': {
        'lassen':   0.157,
        'pascal':   0.365,
        'ray':   0.185,
    }
}

#@todo: add other cluster if need be
vocab_loc = {
    'vast': '/p/vast1/lbann/datasets/atom/enamine_all2018q1_2020q1-2_mpro_inhib_kekulized.vocab',
}
# ==============================================
# Setup LBANN experiment
# ==============================================
def list2str(l):
    return ' '.join(l)

def setup_experiment(lbann, weekly):
    """Construct LBANN experiment.

    Args:
        lbann (module): Module for LBANN Python frontend

    """
    if tools.system(lbann) != 'lassen' and tools.system(lbann) != 'pascal' and tools.system(lbann) != 'ray':
      message = f'{os.path.basename(__file__)} is only supported on lassen, ray, and pascal systems'
      print('Skip - ' + message)
      pytest.skip(message)

    if weekly:
        options = weekly_options_and_targets
    else:
        options = nightly_options_and_targets

    trainer = lbann.Trainer(mini_batch_size=options['mini_batch_size'])
    import atom_models
    pad_index = 40
    sequence_length = 100
    embedding_dim = 42
    dictionary_size = 42
    g_mean = 0.0
    g_std = 1.0
    z_dim = 128
    num_io_threads = 11
    vocab = None
    delimiter = "c"
    warmup = False
    lambda_ = 0.00157
    learn_rate=3e-4

    model = atom_models.construct_atom_wae_model(pad_index=pad_index,
                                                 sequence_length=sequence_length,
                                                 embedding_size=embedding_dim,
                                                 dictionary_size=dictionary_size,
                                                 z_dim=z_dim,
                                                 g_mean=g_mean,
                                                 g_std=g_std,
                                                 lambda_=lambda_,
                                                 lr=learn_rate,
                                                 batch_size=mini_batch_size,
                                                 num_epochs=options['num_epochs'],
                                                 CPU_only=True)

    #see: <LBANN>ci_test/common_python/data/atom/data_reader_mpro.prototext
    import data.atom
    data_reader = data.atom.make_data_reader(lbann)
    # Use less training data for the integration test
    data_reader.reader[0].fraction_of_data_to_use = options['fraction_of_data_to_use']

    opt = lbann.Adam(learn_rate=3e-4, beta1=0.9, beta2=0.99, eps=1e-8)
    return trainer, model, data_reader, opt, options['num_nodes']

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
    input_ = lbann.Input(name='inp', data_field="samples")
    wae_loss= []
    input_feature_dims = sequence_length

    embedding_size = 42
    dictionary_size = 42

    z = lbann.Gaussian(mean=0.0,stdev=1.0, neuron_dims=128)
    recon, d1_real, d1_fake, d_adv, _ = molwae.MolWAE(input_feature_dims,
                                                      dictionary_size,
                                                      embedding_size,
                                                      pad_index,
                                                      num_decoder_layers)(input_,z)

    zero  = lbann.Constant(value=0.0,num_neurons=[1],name='zero')
    one  = lbann.Constant(value=1.0,num_neurons=[1],name='one')

    d1_real_bce = lbann.SigmoidBinaryCrossEntropy([d1_real,one],name='d1_real_bce')
    d1_fake_bce = lbann.SigmoidBinaryCrossEntropy([d1_fake,zero],name='d1_fake_bce')
    d_adv_bce = lbann.SigmoidBinaryCrossEntropy([d_adv,one],name='d_adv_bce')

    wae_loss.append(recon)

    layers = list(lbann.traverse_layer_graph(input_))

    # Hack to avoid non-deterministic floating-point errors in some
    # GPU layers
    for l in layers:
        if isinstance(l, lbann.Embedding) or isinstance(l, lbann.Tessellate):
            l.device = 'CPU'

    # Setup objective function
    weights = set()
    src_layers = []
    dst_layers = []
    for l in layers:
      if(l.weights and "disc0" in l.name and "instance1" in l.name):
        src_layers.append(l.name)
      #freeze weights in disc2
      if(l.weights and "disc1" in l.name and "instance1" in l.name):
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
    def func(cluster, dirname, weekly):

        if weekly:
            targets = weekly_options_and_targets
        else:
            targets = nightly_options_and_targets

        # Run LBANN experiment
        experiment_output = test_func(cluster, dirname, weekly)

        # Parse LBANN log file
        train_recon = None
        test_recon = None
        mini_batch_times = []
        with open(experiment_output['stdout_log_file']) as f:
            for line in f:
                match = re.search('training epoch [0-9]+ recon : ([-0-9.]+)', line)
                if match:
                    train_recon = float(match.group(1))
                match = re.search('test recon : ([-0-9.]+)', line)
                if match:
                    test_recon = float(match.group(1))
                match = re.search('training epoch [0-9]+ mini-batch time statistics : ([-0-9.]+)s mean', line)
                if match:
                    mini_batch_times.append(float(match.group(1)))

        # Check if training reconstruction is within expected range
        assert ((train_recon > targets['expected_train_recon_range'][0]
                 and train_recon < targets['expected_train_recon_range'][1])), \
                f"train reconstruction loss {train_recon:.3f} is outside expected range " + \
                f"[{targets['expected_train_recon_range'][0]:.3f},{targets['expected_train_recon_range'][1]:.3f}]"

        # Check if testing reconstruction  is within expected range
        assert ((test_recon > targets['expected_test_recon_range'][0]
                 and test_recon < targets['expected_test_recon_range'][1])), \
                f"test reconstruction loss {test_recon:.3f} is outside expected range " + \
                f"[{targets['expected_test_recon_range'][0]:.3f},{targets['expected_test_recon_range'][1]:.3f}]"

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

m_lbann_args=f"--vocab={vocab_loc['vast']} --sequence_length=100 --use_data_store --preload_data_store"
# Create test functions that can interact with PyTest
for _test_func in tools.create_tests(setup_experiment,
                                     __file__,
                                     lbann_args=[m_lbann_args]):
    globals()[_test_func.__name__] = augment_test_func(_test_func)
