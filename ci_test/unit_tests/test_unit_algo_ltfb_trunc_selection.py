"""Test to check truncation selection exchanges in LTFB.

An LTFB round is performed after every training step and two (truncation_k=2)
winners chosen from higher random metric values, propagate their
models/topologies to other trainers with lower metric values.
The log files are post-processed to make sure that the correct weights
are propagated by LTFB.

"""
import os
import os.path
import random
import re
import sys

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

# ==============================================
# Setup LBANN experiment
# ==============================================

def setup_experiment(lbann, weekly):
    """Construct LBANN experiment.

    Args:
        lbann (module): Module for LBANN Python frontend

    """

    message = f'{os.path.basename(__file__)} is temporarily failing intermittently on all systems... disable'
    print('Skip - ' + message)
    pytest.skip(message)

    # Setup the training algorithm
    SGD = lbann.BatchedIterativeOptimizer
    TSE = lbann.TruncationSelectionExchange
    metalearning = TSE(
        metric_strategies={'random': TSE.MetricStrategy.HIGHER_IS_BETTER},
        truncation_k=2)
    ltfb = lbann.LTFB("ltfb",
                      metalearning=metalearning,
                      local_algo=SGD("local sgd",
                                     num_iterations=1),
                      metalearning_steps=4)

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
                    sending_partner = [[] for _ in range(num_trainers)]
                    tournament_metrics = [[] for _ in range(num_trainers)]

                #sender
                match = re.search(
                    'In LTFB TSE .* '
                    'trainer ([0-9]+) with score .* sends model to trainer  ([0-9]+) '
                    'with score .*',
                    line)
                if match:
                    trainer = int(match.group(1))
                    sending_partner[trainer].append(trainer) #ltfb_sender

                #receiver
                match = re.search(
                    'In LTFB TSE .* '
                    'trainer ([0-9]+) with score .* receives model from trainer ([0-9]+) '
                    'with score .*',
                    line)
                if match:
                    receiver = loser = trainer = int(match.group(1))
                    sender = winner = partner = int(match.group(2))
                    sending_partner[trainer].append(sender) #ltfb_sender

                # Metric value on tournament set
                match = re.search(
                    'model0 \\(instance ([0-9]+)\\) tournament random : '
                    '([0-9.]+)',
                    line)
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
        #@todo, add more checks

        # Make sure metric values match expected values
        # All trainers participate in tournament by evaluating their local
        # model on tournament dataset
        # Winning trainers (above threshold) retain their models
        # Losing trainers (below threshold) receive models from winning trainers
        # Here we test that the model exchanges between winners and lossers are correct
        for step in range(_num_epochs-1):
            for trainer in range(num_trainers):
                if (len(sending_partner[trainer]) != 0):
                  sender_at_step = sending_partner[trainer][step]
                  trainer_score = tournament_metrics[trainer][step]
                  winning_score = tournament_metrics[sender_at_step][step]

                  assert trainer_score <= winning_score, \
                      'Incorrect metric value for LTFB tournament'

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
