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

# ==============================================
# Options
# ==============================================

# Training options
num_epochs = 5
mini_batch_size = 256
num_nodes = 2
imagenet_fraction = 0.0009992   # Train with 1280 out of 1.28M samples

# Top-1 classificaiton accuracy
expected_train_accuracy_range = (1.5, 3.5)

# Average mini-batch time (in sec) for each LC system
expected_mini_batch_times = {
    'pascal': 0.15,
    'catalyst': 6.0,
    'lassen': -1,
    'corona': 28.5
}

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
    optimizer = lbann.SGD(learn_rate=0.1, momentum=0.9)
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
    images = lbann.Identity(input_)
    labels = lbann.Identity(input_)
    x = lbann.models.ResNet18(1000, bn_statistics_group_size=-1)(images)
    probs = lbann.Softmax(x)
    cross_entropy = lbann.CrossEntropy([probs, labels])
    accuracy = lbann.CategoricalAccuracy([probs, labels])
    layers = list(lbann.traverse_layer_graph(x))

    # Setup objective function
    l2_reg_weights = set()
    for l in layers:
        if type(l) == lbann.Convolution or type(l) == lbann.FullyConnected:
            l2_reg_weights.update(l.weights)
    l2_reg = lbann.L2WeightRegularization(weights=l2_reg_weights, scale=1e-4)
    obj = lbann.ObjectiveFunction([cross_entropy, l2_reg])

    # Objects for LBANN model
    callbacks = [lbann.CallbackPrint(), lbann.CallbackTimer()]
    metrics = [lbann.Metric(accuracy, name='accuracy', unit='%')]

    # Construct model
    return lbann.Model(mini_batch_size,
                       num_epochs,
                       layers=layers,
                       objective_function=obj,
                       metrics=metrics,
                       callbacks=callbacks)

def construct_data_reader(lbann):
    """Construct Protobuf message for Python data reader.

    The Python data reader will import the current Python file to
    access the sample access functions.

    Args:
        lbann (module): Module for LBANN Python frontend

    """

    # TODO (tym): Figure out how to switch between LBANN builds. See
    # GitHub Issue #1289.
    import lbann.contrib.lc.paths

    # Load data readers from prototext
    dirname = os.path.dirname
    lbann_dir = dirname(dirname(dirname(os.path.realpath(__file__))))
    pb_file = os.path.join(lbann_dir,
                           'model_zoo',
                           'data_readers',
                           'data_reader_imagenet.prototext')
    message = lbann.lbann_pb2.LbannPB()
    with open(pb_file, 'r') as f:
        google.protobuf.text_format.Merge(f.read(), message)
    message = message.data_reader

    # Only use data reader for training set
    del message.reader[1:]

    # We want to train with 1280 out of 1.28M samples in ImageNet-1K
    message.reader[0].percent_of_data_to_use = imagenet_fraction

    # Set location of ImageNet-1K data
    message.reader[0].data_filedir = lbann.contrib.lc.paths.imagenet_dir()
    message.reader[0].data_filename = lbann.contrib.lc.paths.imagenet_labels()

    return message

# ==============================================
# Setup PyTest
# ==============================================

# Create test name by removing '.py' from file name
current_file = os.path.realpath(__file__)
_test_name = os.path.splitext(os.path.basename(current_file))[0]

# Basic test function
def _test_func(cluster, executables, dir_name, compiler_name):
    tools.process_executable(_test_name, compiler_name, executables)

    # Choose LBANN build and load Python frontend
    if compiler_name == 'exe':
        exe = executables[compiler_name]
        bin_dir = os.path.dirname(exe)
        install_dir = os.path.dirname(bin_dir)
        build_path = '{i}/lib/python3.7/site-packages'.format(i=install_dir)
    else:
        if compiler_name == 'clang6':
            path = 'clang.Release'
        elif compiler_name == 'clang6_debug':
            path = 'clang.Debug'
        elif compiler_name == 'gcc7':
            path = 'gnu.Release'
        elif compiler_name == 'clang6_debug':
            path = 'gnu.Debug'
        elif compiler_name == 'intel19':
            path = 'intel.Release'
        elif compiler_name == 'intel19_debug':
            path = 'intel.Debug'
        path = '{p}.{c}.llnl.gov'.format(p=path, c=cluster)
        build_path = '{d}/build/{p}/install/lib/python3.7/site-packages'.format(
            d=dir_name, p=path)
    print('build_path={b}'.format(b=build_path))
    sys.path.append(build_path)
    import lbann
    import lbann.contrib.lc.launcher

    # Setup LBANN experiment
    trainer, model, data_reader, optimizer = setup_experiment(lbann)

    # Run LBANN experiment
    kwargs = {
        'nodes': num_nodes,
        'overwrite_script': True
    }
    result_dir = os.path.join(os.path.dirname(current_file),
                              'experiments',
                              '{}_{}'.format(_test_name, compiler_name))
    stdout_file_name = os.path.join(result_dir, 'out.log')
    stderr_file_name = os.path.join(result_dir, 'err.log')
    return_code = lbann.contrib.lc.launcher.run(
        trainer=trainer,
        model=model,
        data_reader=data_reader,
        optimizer=optimizer,
        experiment_dir=result_dir,
        job_name='lbann_{}'.format(_test_name),
        **kwargs)
    tools.assert_success(return_code, stderr_file_name)

    # Parse LBANN log file
    train_accuracies = []
    mini_batch_times = []
    with open(stdout_file_name) as f:
        for line in f:
            match = re.search('training epoch [0-9]+ accuracy : ([0-9.]+)%', line)
            if match:
                train_accuracies.append(float(match.group(1)))
            match = re.search('training epoch [0-9]+ mini-batch time statistics : ([0-9.]+)s mean', line)
            if match:
                mini_batch_times.append(float(match.group(1)))

    # Check if training accuracy is within expected range
    train_accuracy = train_accuracies[-1]
    assert (expected_train_accuracy_range[0]
            < train_accuracy
            < expected_train_accuracy_range[1]), \
            'train accuracy is outside expected range'

    # Check if mini-batch time is within expected range
    # Note: Skip first epoch since its runtime is usually an outlier
    mini_batch_times = mini_batch_times[1:]
    mini_batch_time = sum(mini_batch_times) / len(mini_batch_times)
    assert (0.75 * expected_mini_batch_times[cluster]
            < mini_batch_time
            < 1.25 * expected_mini_batch_times[cluster]), \
            'average mini-batch time is outside expected range'

# Specific test functions for different build configurations
def _test_func_exe(cluster, dirname, exe):
    if exe is None:
        e = '{}_exe: Non-local testing'.format(_test_name)
        print('Skip - ' + e)
        pytest.skip(e)
    exes = {'exe': exe}
    _test_func(cluster, exes, dirname, 'exe')
def _test_func_clang6(cluster, exes, dirname):
    _test_func(cluster, exes, dirname, 'clang6')
def _test_func_gcc7(cluster, exes, dirname):
    _test_func(cluster, exes, dirname, 'gcc7')
def _test_func_intel19(cluster, exes, dirname):
    _test_func(cluster, exes, dirname, 'intel19')
globals()['{}_exe'.format(_test_name)] = _test_func_exe
globals()['{}_clang6'.format(_test_name)] = _test_func_clang6
globals()['{}_gcc7'.format(_test_name)] = _test_func_gcc7
globals()['{}_intel19'.format(_test_name)] = _test_func_intel19
