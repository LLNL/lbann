"""Test to check truncation selection exchanges in LTFB.

An LTFB round is performed after every training step and two (truncation_k=2)
winners chosen from higher random metric values, propagate their
models/topologies to other trainers with lower metric values.
The log files are post-processed to make sure that the correct weights
are propagated by LTFB.

"""
from collections import defaultdict
import os
import os.path
import random
import re
import sys
import pytest

# Bamboo utilities
current_file = os.path.realpath(__file__)
current_dir = os.path.dirname(current_file)
sys.path.insert(0, os.path.join(os.path.dirname(current_dir), 'common_python'))
import tools

# ==============================================
# Objects for Python data reader
# ==============================================
# Note: The Python data reader imports this file as a module and calls
# the functions below to ingest data.

# RNG
rng_pid = None
def initialize_rng():
    """Initialize random seed if needed.

    Seed should be initialized independently on each process. We
    reinitialize if we detect a process fork.

    """
    global rng_pid
    if rng_pid != os.getpid():
        rng_pid = os.getpid()
        random.seed()

# Sample access functions
_mini_batch_size = 2
_num_epochs = 5
def get_sample(index):
    initialize_rng()
    return (random.uniform(0,1),)
def num_samples():
    return _mini_batch_size
def sample_dims():
    return (1,)

# Meta-learning parameters
_top_k = 2
_metalearning_steps = 4

# ==============================================
# Setup LBANN experiment
# ==============================================

def setup_experiment(lbann, weekly):
    """Construct LBANN experiment.

    Args:
        lbann (module): Module for LBANN Python frontend

    """
    # Setup the training algorithm
    SGD = lbann.BatchedIterativeOptimizer
    TSE = lbann.TruncationSelectionExchange
    metalearning = TSE(
        metric_strategies={'random': TSE.MetricStrategy.HIGHER_IS_BETTER},
        truncation_k=_top_k)
    ltfb = lbann.LTFB("ltfb",
                      metalearning=metalearning,
                      local_algo=SGD("local sgd",
                                     num_iterations=1),
                      metalearning_steps=_metalearning_steps)

    trainer = lbann.Trainer(_mini_batch_size,
                            training_algo=ltfb)
    model = construct_model(lbann)
    data_reader = construct_data_reader(lbann)
    optimizer = lbann.NoOptimizer()
    return trainer, model, data_reader, optimizer, None # Don't request any specific number of nodes

def construct_model(lbann):
    """Construct LBANN model.

    Args:
        lbann (module): Module for LBANN Python frontend

    """

    # Layer graph
    rand = lbann.Input(data_field='samples')
    layers = list(lbann.traverse_layer_graph([rand]))
    for l in layers:
        l.device = 'CPU'

    # Model objects
    metrics = [
        lbann.Metric(rand, name='random'),
    ]
    callbacks = [
        lbann.CallbackPrint(),
    ]

    # Construct model
    return lbann.Model(_num_epochs,
                       layers=layers,
                       metrics=metrics,
                       callbacks=callbacks)

def construct_data_reader(lbann):
    """Construct Protobuf message for Python data reader.

    The Python data reader will import the current Python file to
    access the sample access functions.

    Args:
        lbann (module): Module for LBANN Python frontend

    """

    # Note: The training data reader should be removed when
    # https://github.com/LLNL/lbann/issues/1098 is resolved.
    message = lbann.reader_pb2.DataReader()
    message.reader.extend([
        tools.create_python_data_reader(
            lbann,
            current_file,
            'get_sample',
            'num_samples',
            'sample_dims',
            'train',
        ),
        tools.create_python_data_reader(
            lbann,
            current_file,
            'get_sample',
            'num_samples',
            'sample_dims',
            'tournament',
        ),
    ])
    return message

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
        num_trainers = None
        log_file = experiment_output['stdout_log_file']
        with open(log_file) as f:
            for line in f:
                # Configure data once we figure out number of trainers
                if num_trainers is None:
                    match = re.search('Trainers *: ([0-9]+)', line)
                    if match:
                        num_trainers = int(match.group(1))
                    else:
                        continue
                    if num_trainers <= _top_k:
                        # There will be no truncation-selection exchanges
                        return
                    sending_partner = [defaultdict(list) for _ in range(num_trainers)]
                    receiving_partner = [
                        defaultdict(list) for _ in range(num_trainers)
                    ]
                    tournament_metrics = [[] for _ in range(num_trainers)]

                #sender
                match = re.search(
                    'In LTFB TSE step ([0-9]+), '
                    'trainer ([0-9]+) with score .* sends model to trainer  ([0-9]+) '
                    'with score .*', line)
                if match:
                    step = int(match.group(1))
                    sender = winner = trainer = int(match.group(2))
                    receiver = loser = partner = int(match.group(3))
                    sending_partner[trainer][step].append(partner)

                #receiver
                match = re.search(
                    'In LTFB TSE step ([0-9]+), '
                    'trainer ([0-9]+) with score .* receives model from trainer ([0-9]+) '
                    'with score .*', line)
                if match:
                    step = int(match.group(1))
                    receiver = loser = trainer = int(match.group(2))
                    sender = winner = partner = int(match.group(3))
                    receiving_partner[trainer][step].append(partner)

                # Metric value on tournament set
                match = re.search(
                    'model0 \\(instance ([0-9]+)\\) tournament random : '
                    '([0-9.]+)', line)
                if match:
                    trainer = int(match.group(1))
                    tournament_metrics[trainer].append(float(match.group(2)))

        # Make sure file has been parsed correctly
        assert num_trainers, \
            f'Error parsing {log_file} (could not find number of trainers)'
        for trainer, vals in enumerate(tournament_metrics):
            assert len(vals) == _num_epochs-1, \
                f'Error parsing {log_file} ' \
                f'(expected {_num_epochs} tournament metric values, ' \
                f'but found {len(vals)} for trainer {trainer})'

        # Make sure the steps executed match the metalearning steps
        steps = 0
        for trainer in range(num_trainers):
            if sending_partner[trainer]:
                steps = max(steps, *sending_partner[trainer].keys())
            if receiving_partner[trainer]:
                steps = max(steps, *receiving_partner[trainer].keys())

        steps += 1  # Log is 0-based
        assert steps == _metalearning_steps, (
            f'Steps captured in log ({steps}) mismatch meta-learning steps '
            f'({_metalearning_steps})')

        # Make sure the sends were executed to the right receivers
        for step in range(steps):
            for trainer in range(num_trainers):
                # Trainer does not participate
                if len(sending_partner[trainer]) + len(
                        receiving_partner[trainer]) == 0:
                    continue

                if step in sending_partner[trainer]:  # If trainer won during that step
                    # Check partner match
                    sending_partners_at_step = sending_partner[trainer][step]
                    for partner in sending_partners_at_step:
                        assert step in receiving_partner[partner], (
                            f'Trainer {partner} did not receive a sent metric from '
                            f'{trainer} during step {step}')
                        assert receiving_partner[partner][step][0] == trainer, (
                            f'Trainer {partner} receive partner mismatch (expected '
                            f'{trainer}, got {receiving_partner[partner][step][0]}) '
                            f'during step {step}')
                else:
                    # Check validity of losing receivers
                    assert step in receiving_partner[trainer], (
                        f'Trainer {trainer} did not send nor receive metric during '
                        f'step {step}')
                    assert len(receiving_partner[trainer][step]) == 1, (
                        f'Trainer {trainer} received from more than one sender at '
                        f'step {step}')

        # Make sure metric values match expected values
        # All trainers participate in tournament by evaluating their local
        # model on tournament dataset
        # Winning trainers (above threshold) retain their models
        # Losing trainers (below threshold) receive models from winning trainers
        # Here we test that the model exchanges between winners and losers are correct
        for step in range(_num_epochs - 1):
            for trainer in range(num_trainers):
                if step in receiving_partner[trainer]:
                    sender_at_step = receiving_partner[trainer][step][0]
                    trainer_score = tournament_metrics[trainer][step]
                    winning_score = tournament_metrics[sender_at_step][step]

                    assert trainer_score <= winning_score, (
                        f'Incorrect metric value for LTFB tournament: step {step}, '
                        f'trainer {trainer} with score {trainer_score}, sender '
                        f'{sender_at_step} with score {winning_score}')


    # Return test function from factory function
    func.__name__ = test_name
    return func

# Create test functions that can interact with PyTest
for _test_func in tools.create_tests(
        setup_experiment,
        __file__,
        nodes=2,
        lbann_args='--procs_per_trainer=2'):
    globals()[_test_func.__name__] = augment_test_func(_test_func)
