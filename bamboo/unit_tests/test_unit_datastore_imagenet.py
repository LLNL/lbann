import os.path
import re
import sys
import math
import numpy as np
import pytest

# Local files
current_file = os.path.realpath(__file__)
current_dir = os.path.dirname(current_file)
sys.path.insert(0, os.path.join(os.path.dirname(current_dir), 'common_python'))
import tools

# ==============================================
# Options
# ==============================================

# Training options
num_epochs = 5
mini_batch_size = 256
num_nodes = 2
imagenet_fraction = 0.0031971 # Train with 4096 out of 1.28M samples
random_seed = 20191206

# ==============================================
# Setup LBANN experiment
# ==============================================

def setup_experiment(lbann):
    """Construct LBANN experiment.

    Args:
        lbann (module): Module for LBANN Python frontend

    """
    trainer = lbann.Trainer()
    model = construct_model(lbann)
    data_reader = construct_data_reader(lbann)
    optimizer = lbann.SGD(learn_rate=0.01, momentum=0.9)
    return trainer, model, data_reader, optimizer

def construct_model(lbann):
    """Construct LBANN model.

    Args:
        lbann (module): Module for LBANN Python frontend

    """

    # TODO (tym): Figure out how to switch between LBANN builds. See
    # GitHub Issue #1289.
    import lbann.models

    # Layer graph
    input_ = lbann.Input()
    x = lbann.Identity(input_)
    y = lbann.L2Norm2(x)
    z = lbann.Multiply(y, lbann.Sqrt(lbann.MiniBatchIndex()))

    # Make sure all layers are on CPU
    for layer in lbann.traverse_layer_graph(input_):
        layer.device = 'cpu'

    # Objects for LBANN model
    callbacks = [lbann.CallbackPrint(), lbann.CallbackTimer()]
    metrics = [lbann.Metric(z, name='metric')]

    # Construct model
    return lbann.Model(mini_batch_size,
                       num_epochs,
                       layers=lbann.traverse_layer_graph(input_),
                       metrics=metrics,
                       callbacks=callbacks,
                       random_seed=random_seed)

def construct_data_reader(lbann):
    """Construct Protobuf message for ImageNet data reader.

    Args:
        lbann (module): Module for LBANN Python frontend

    """

    # TODO (tym): Figure out how to switch between LBANN builds. See
    # GitHub Issue #1289.
    import lbann.contrib.lc.paths

    # Construct data reader
    message = lbann.reader_pb2.DataReader()
    reader = message.reader.add()

    # Configure data reader
    reader.name = 'imagenet'
    reader.role = 'train'
    reader.data_filedir = lbann.contrib.lc.paths.imagenet_dir(data_set='train')
    reader.data_filename = lbann.contrib.lc.paths.imagenet_labels(data_set='train')
    reader.percent_of_data_to_use = imagenet_fraction
    reader.num_labels = 1000
    reader.shuffle = True

    # Configure transforms
    # Note: The image just resized to 32x32
    resize = reader.transforms.add().resize
    resize.SetInParent()
    resize.height = 32
    resize.width = 32
    colorize = reader.transforms.add().colorize
    colorize.SetInParent()
    normalize = reader.transforms.add().to_lbann_layout
    normalize.SetInParent()

    return message

# ==============================================
# Setup PyTest
# ==============================================

def create_test_func(baseline_test_func, datastore_test_func):
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
    test_name = datastore_test_func.__name__

    # Define test function
    def func(cluster, exes, dirname, weekly):

        # Run LBANN experiment without data store
        baseline_test_output = baseline_test_func(cluster, exes, dirname)
        baseline_metrics = []
        with open(baseline_test_output['stdout_log_file']) as f:
            for line in f:
                match = re.search('training epoch [0-9]+ metric : ([0-9.]+)', line)
                if match:
                    baseline_metrics.append(float(match.group(1)))

        # Run LBANN experiment with data store
        datastore_test_output = datastore_test_func(cluster, exes, dirname)
        datastore_metrics = []
        with open(datastore_test_output['stdout_log_file']) as f:
            for line in f:
                match = re.search('training epoch [0-9]+ metric : ([0-9.]+)', line)
                if match:
                    datastore_metrics.append(float(match.group(1)))

        # Check if metrics are same in baseline and data store experiments
        # Note: "Print statistics" callback will print up to 6 digits
        # of metric values.
        assert len(baseline_metrics) == len(datastore_metrics), \
            'baseline and data store experiments did not run for same number of epochs'
        for i in range(len(datastore_metrics)):
            x = baseline_metrics[i]
            xhat = datastore_metrics[i]
            eps = np.finfo(np.float32).eps
            ceillogx = int(math.ceil(math.log10(x)))
            assert abs(x-xhat) < max(8*eps*x, 1.5*10**(ceillogx-6)), \
                'found large discrepancy in metrics for baseline and data store experiments'

    # Return test function from factory function
    func.__name__ = test_name
    return func

# Create test functions that can interact with PyTest
baseline_tests = tools.create_tests(
    setup_experiment,
    __file__,
    test_name_base='test_unit_datastore_imagenet_nodatastore',
    nodes=num_nodes
)
datastore_tests = tools.create_tests(
    setup_experiment,
    __file__,
    nodes=num_nodes,
    lbann_args=['--use_data_store']
)
for i in range(len(datastore_tests)):
    _test_func = create_test_func(baseline_tests[i], datastore_tests[i])
    globals()[_test_func.__name__] = _test_func
