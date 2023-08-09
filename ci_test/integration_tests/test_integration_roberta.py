import functools
import operator
import os
import os.path
import re
import sys
from types import SimpleNamespace
import numpy as np
import warnings
import pytest

# Local files
current_file = os.path.realpath(__file__)
current_dir = os.path.dirname(current_file)
sys.path.insert(0, os.path.join(os.path.dirname(current_dir), "common_python"))
import tools
import data.iur

# ==============================================
# Options
# ==============================================

# Model params
config = SimpleNamespace()
config.attention_probs_dropout_prob = 0.0
config.hidden_act = "gelu"
config.hidden_dropout_prob = 0.0
config.hidden_size = 768
config.input_shape = (16, 32)
config.intermediate_size = 3072
config.layer_norm_eps = 1e-05
config.max_position_embeddings = 514
config.num_attention_heads = 12
config.num_classes = 1000
config.num_hidden_layers = 6
config.pad_token_id = 0
config.position_embedding_type = "absolute"
config.type_vocab_size = 1
config.vocab_size = 50265
weights_dir = "/p/vast1/lbann/pretrained_weights/RoBERTa/"

# Training options

# Average mini-batch time (in sec) for each LC system
# Note that run times are with LBANN_DETERMINISTIC set

################################################################################
# Weekly training options and targets
################################################################################
weekly_options_and_targets = {
    'num_nodes': 1,
    'num_epochs': 0,
    'mini_batch_size': 16,
    # Test loss range expanded from [6.8, 7.1] to [6.1, 7.2]
    'expected_test_loss_range': (6.1, 7.2),
    'fraction_of_data_to_use': 1.0,
    'expected_mini_batch_times': {
        "pascal": 0.1225,
        "lassen": 0.0440,
        "ray" : 0.0607,
        "catalyst" : 7.45,
    }
}

################################################################################
# Nightly training options and targets
################################################################################
nightly_options_and_targets = {
    'num_nodes': 1,
    'num_epochs': 0,
    'mini_batch_size': 16,
    # Test loss range expanded from [6.75, 7.1] to [6.1, 7.2]
    'expected_test_loss_range': (6.1, 7.2),
    'fraction_of_data_to_use': 0.01,
    'expected_mini_batch_times': {
        "pascal": 0.925, # Weird performance behavior 3/21/2022 - 0.1225,
        "lassen": 1.409, # BVE Changed again on 9/21/22 from 0.808 # Weird performance regression 3/21/2022 - 0.0440,
        "ray" : 0.578,
        "catalyst" : 7.45,
        "tioga":  0.925, # BVE dummy value from pascal
        "corona": 0.925, # BVE dummy value from pascal
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
    if weekly:
        options = weekly_options_and_targets
    else:
        options = nightly_options_and_targets

    trainer = lbann.Trainer(mini_batch_size=options['mini_batch_size'])
    model = construct_model(lbann, options['num_epochs'])

    data_reader = data.iur.make_data_reader(lbann)
    data_reader.reader[0].fraction_of_data_to_use = options['fraction_of_data_to_use']

    optimizer = lbann.Adam(learn_rate=1e-3, beta1=0.9, beta2=0.99, eps=1e-8)
    return trainer, model, data_reader, optimizer, options['num_nodes']


def construct_model(lbann, num_epochs):
    """Construct LBANN model.

    Args:
        lbann (module): Module for LBANN Python frontend

    """

    import lbann.models

    # Layer graph
    input_ = lbann.Slice(
        lbann.Input(data_field="samples"),
        slice_points=[0, 1, 1 + np.prod(config.input_shape)],
    )
    labels = lbann.Identity(input_)
    samples = lbann.Reshape(input_, dims=config.input_shape)
    x = lbann.models.RoBERTa(config, load_weights=weights_dir)(samples)
    log_probs = lbann.LogSoftmax(
        lbann.FullyConnected(x, num_neurons=config.num_classes, has_bias=False)
    )
    label_onehot = lbann.OneHot(labels, size=config.num_classes)
    loss = lbann.Negative(
        lbann.Reduction(lbann.Multiply(log_probs, label_onehot), mode="sum")
    )

    # Objects for LBANN model
    callbacks = [lbann.CallbackPrint(), lbann.CallbackTimer()]
    metrics = [lbann.Metric(loss, name="loss")]

    # Construct model
    return lbann.Model(
        num_epochs,
        layers=lbann.traverse_layer_graph(input_),
        objective_function=loss,
        metrics=metrics,
        callbacks=callbacks,
    )


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
        test_loss = None
        mini_batch_time = None
        with open(experiment_output["stdout_log_file"]) as f:
            for line in f:
                match = re.search("test loss : ([0-9.]+)", line)
                if match:
                    test_loss = float(match.group(1))
                match = re.search(
                    "test mini-batch time statistics : ([0-9.]+)s mean", line
                )
                if match:
                    mini_batch_time = float(match.group(1))

        # Check if testing accuracy is within expected range
        assert ((test_loss > targets['expected_test_loss_range'][0]
                 and test_loss < targets['expected_test_loss_range'][1])), \
                f"test loss {test_loss:.3f} is outside expected range " + \
                f"[{targets['expected_test_loss_range'][0]:.3f},{targets['expected_test_loss_range'][1]:.3f}]"

        # Check if mini-batch time is within expected range
        # Note: Skip first epoch since its runtime is usually an outlier
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
for _test_func in tools.create_tests(setup_experiment, __file__, time_limit=3):
    globals()[_test_func.__name__] = augment_test_func(_test_func)
