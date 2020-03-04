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
num_epochs = 1
#XX num_epochs = 5
mini_batch_size = 256
num_nodes = 2
imagenet_fraction = 0.0031971 # Train with 4096 out of 1.28M samples
random_seed = 20191206
base_test_name='test_unit_datastore_imagenet_'

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
def create_datastore_test_func(test_func, baseline_metrics, cluster, exes, dirname) :
    r = [test_func.__name__]
    datastore_test_output = test_func(cluster, exes, dirname)
    datastore_metrics = []
    with open(datastore_test_output['stdout_log_file']) as f:
        for line in f:
            match = re.search('training epoch [0-9]+ metric : ([0-9.]+)', line)
            if match:
                datastore_metrics.append(float(match.group(1)))

    # Check if metrics are same in baseline and data store experiments
    # Note: "Print statistics" callback will print up to 6 digits
    # of metric values.
    if len(baseline_metrics) == len(datastore_metrics) :
        r.append(test_func.__name__ + ' :: baseline and data store experiments did not run for same number of epochs')
        return r

    for i in range(len(datastore_metrics)):
        x = baseline_metrics[i]
        xhat = datastore_metrics[i]
        eps = np.finfo(np.float32).eps
        ceillogx = int(math.ceil(math.log10(x)))
        if abs(x-xhat) < max(8*eps*x, 1.5*10**(ceillogx-6)) :
            return(test_func.__name__, ' :: found large discrepancy in metrics for baseline and data store experiments')

    # TODO:
    # Check if the output 'data_store_profile_train.txt' file
    # contains the entries specified in the 'profile_tests' input param

    return r

def create_test_func(baseline_test_func, datastore_test_funcs) :
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
    test_name = baseline_test_func.__name__

    # Define test function
    def func(cluster, exes, dirname, weekly):
        num_failed = 0
        results = []

        # Run LBANN experiment without data store
        baseline_test_output = baseline_test_func(cluster, exes, dirname)
        baseline_metrics = []
        with open(baseline_test_output['stdout_log_file']) as f:
            for line in f:
                match = re.search('training epoch [0-9]+ metric : ([0-9.]+)', line)
                if match:
                    baseline_metrics.append(float(match.group(1)))

        # Run LBANN experiments with data store
        for i in range(len(datastore_test_funcs)) :
            r = create_datastore_test_func(datastore_test_funcs[i], baseline_metrics, cluster, exes, dirname)
            results.append(r)
            if len(r) > 1 :
              num_failed += 1


        assert num_failed == 0, 'num tests failed: ' + str(num_failed)
    #assert all_tests_passed, '\n' + ' '.join(results)

    # Return test function from factory function
    func.__name__ = test_name
    return func
 
# Create test functions that can interact with PyTest
def make_test(name, test_by_platform_list=[], args=[]) :
    test_list = tools.create_tests(
            setup_experiment,
            __file__,
            nodes=num_nodes,
            test_name_base='test_unit_datastore_imagenet_' + name,
            lbann_args=args)
    if test_by_platform_list != [] :
        for i in range(len(test_list)) :
          test_by_platform_list[i].append(test_list[i])
    else :
      return test_list

baseline_tests = make_test('nodatastore')
datastore_tests = [[] for j in range(len(baseline_tests))]

#local cache with explicit loading
ds = make_test('data_store_cache_explicit', datastore_tests, ['--data_store_cache', '--data_store_profile'])

#local cache with preloading
ds = make_test('data_store_cache_preloading', datastore_tests, ['--data_store_cache', '--preload_data_store', '--data_store_profile'])

#local cache with preloading -- this will fail; only for use
#during test development
##ds = make_test('data_store_cache_preloading', datastore_tests, ['--data_store_cache', '--preload_data_store', '--data_store_profile', '--data_store_fail'])

for i in range(len(datastore_tests)):
    _test_func = create_test_func(baseline_tests[i], datastore_tests[i])
    globals()[_test_func.__name__] = _test_func
