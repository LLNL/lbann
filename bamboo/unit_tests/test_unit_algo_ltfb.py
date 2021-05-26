"""Test to check weight exchanges in LTFB.

Each model has a randomly initialized weights object. An LTFB round is
performed after every training step and winners are chosen randomly.
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
    return (random.gauss(0,1),)
def num_samples():
    return _mini_batch_size
def sample_dims():
    return (1,)

# ==============================================
# Setup LBANN experiment
# ==============================================

def setup_experiment(lbann):
    """Construct LBANN experiment.

    Args:
        lbann (module): Module for LBANN Python frontend

    """

    # Setup the training algorithm
    RPE = lbann.RandomPairwiseExchange
    SGD = lbann.BatchedIterativeOptimizer
    metalearning = RPE(
        metric_strategies={'random': RPE.MetricStrategy.HIGHER_IS_BETTER})
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
    return trainer, model, data_reader, optimizer

def construct_model(lbann):
    """Construct LBANN model.

    Args:
        lbann (module): Module for LBANN Python frontend

    """

    # Layer graph
    weight = lbann.Weights(initializer=lbann.UniformInitializer(min=0, max=1))
    weight = lbann.WeightsLayer(weights=weight, dims=tools.str_list([1]))
    rand = lbann.Identity(lbann.Input())
    layers = list(lbann.traverse_layer_graph([weight, rand]))
    for l in layers:
        l.device = 'CPU'

    # Model objects
    metrics = [
        lbann.Metric(weight, name='weight'),
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
            'validate',
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
    def func(cluster, dirname):

        # Run LBANN experiment
        experiment_output = test_func(cluster, dirname)

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
                    ltfb_partners = [[] for _ in range(num_trainers)]
                    ltfb_winners = [[] for _ in range(num_trainers)]
                    tournament_metrics = [[] for _ in range(num_trainers)]
                    validation_metrics = [[] for _ in range(num_trainers)]

                # LTFB tournament winners
                match = re.search(
                    'LTFB .* '
                    'trainer ([0-9]+) selected model from trainer ([0-9]+) '
                    '\\(trainer [0-9]+ score .* trainer ([0-9]+) score.*\\)',
                    line)
                if match:
                    trainer = int(match.group(1))
                    winner = int(match.group(2))
                    partner = int(match.group(3))
                    ltfb_partners[trainer].append(partner)
                    ltfb_winners[trainer].append(winner)

                # Metric value on tournament set
                match = re.search(
                    'model0 \\(instance ([0-9]+)\\) tournament weight : '
                    '([0-9.]+)',
                    line)
                if match:
                    trainer = int(match.group(1))
                    tournament_metrics[trainer].append(float(match.group(2)))

                # Metric value on validation set
                match = re.search(
                    'model0 \\(instance ([0-9]+)\\) validation weight : '
                    '([0-9.]+)',
                    line)
                if match:
                    trainer = int(match.group(1))
                    validation_metrics[trainer].append(float(match.group(2)))

        # Make sure file has been parsed correctly
        assert num_trainers, \
            f'Error parsing {log_file} (could not find number of trainers)'
        for trainer, partners in enumerate(ltfb_partners):
            assert len(partners) == _num_epochs-1, \
                f'Error parsing {log_file} ' \
                f'(expected {_num_epochs-1} LTFB rounds, ' \
                f'but found {len(partners)} for trainer {trainer})'
        for trainer, winners in enumerate(ltfb_winners):
            assert len(winners) == _num_epochs-1, \
                f'Error parsing {log_file} ' \
                f'(expected {_num_epochs-1} LTFB rounds, ' \
                f'but found {len(winners)} for trainer {trainer})'
        for trainer, vals in enumerate(validation_metrics):
            assert len(vals) == _num_epochs, \
                f'Error parsing {log_file} ' \
                f'(expected {_num_epochs} validation metric values, ' \
                f'but found {len(vals)} for trainer {trainer})'
        for trainer, vals in enumerate(tournament_metrics):
            assert len(vals) == 2*(_num_epochs-1), \
                f'Error parsing {log_file} ' \
                f'(expected {_num_epochs} validation metric values, ' \
                f'but found {len(vals)} for trainer {trainer})'

        # Make sure metric values match expected values
        # Note: An LTFB round occurs once per training epoch
        # (excluding the first epoch). Each LTFB round involves two
        # evaluations on the tournament set: once on the local model
        # and once on a model from a partner trainer. At the end of
        # each training epoch, we perform an evalutation on the
        # validation set. By inspecting the metric values
        # (corresponding to the model weight), we can make sure that
        # LTFB is evaluating on the correct models.
        tol = 1e-4
        for step in range(_num_epochs-1):
            for trainer in range(num_trainers):
                partner = ltfb_partners[trainer][step]
                winner = ltfb_winners[trainer][step]
                local_val = tournament_metrics[trainer][2*step]
                partner_val = tournament_metrics[trainer][2*step+1]
                winner_val = validation_metrics[trainer][step+1]
                true_local_val = validation_metrics[trainer][step]
                true_partner_val = validation_metrics[partner][step]
                true_winner_val = validation_metrics[winner][step]
                assert true_local_val-tol < local_val < true_local_val+tol, \
                    'Incorrect metric value for LTFB local model'
                assert true_partner_val-tol < partner_val < true_partner_val+tol, \
                    'Incorrect metric value for LTFB partner model'
                assert true_winner_val-tol < winner_val < true_winner_val+tol, \
                    'Incorrect metric value for LTFB winner model'

    # Return test function from factory function
    func.__name__ = test_name
    return func

# Create test functions that can interact with PyTest
for _test_func in tools.create_tests(setup_experiment,
                               __file__,
                               nodes=2,
                               lbann_args='--procs_per_trainer=2'):
    globals()[_test_func.__name__] = augment_test_func(_test_func)
