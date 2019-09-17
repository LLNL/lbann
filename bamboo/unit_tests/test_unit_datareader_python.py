import os
import os.path
import sys
import numpy as np
import pytest

# Local files
current_file = os.path.realpath(__file__)
current_dir = os.path.dirname(current_file)
sys.path.insert(0, os.path.join(os.path.dirname(current_dir), 'common_python'))
import tools

# ==============================================
# Objects for Python data reader
# ==============================================
# Note: The Python data reader imports this file and calls the
# functions below to ingest data. This is the only part of the script
# that should be executed when the script is imported, or else the
# Python data reader might misbehave.

# Data
np.random.seed(20190708)
_num_samples = 23
_sample_size = 7
_samples = np.random.normal(size=(_num_samples,_sample_size))
_samples = _samples.astype(np.float32)

# Sample access functions
def get_sample(index):
    return _samples[index,:]
def num_samples():
    return _num_samples
def sample_dims():
    return (_sample_size,)

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
    optimizer = lbann.NoOptimizer()
    return trainer, model, data_reader, optimizer

def construct_model(lbann):
    """Construct LBANN model.

    Args:
        lbann (module): Module for LBANN Python frontend

    """

    # Layer graph
    x = lbann.Input()
    obj = lbann.L2Norm2(x)
    layers = list(lbann.traverse_layer_graph(x))
    metric = lbann.Metric(obj, name='obj')
    callbacks = []

    # Compute expected value with NumPy
    vals = []
    for i in range(num_samples()):
        x = get_sample(i)
        obj = np.inner(x, x)
        vals.append(obj)
    val = np.mean(vals)
    tol = 8 * val * np.finfo(np.float32).eps
    callbacks.append(lbann.CallbackCheckMetric(
        metric=metric.name,
        lower_bound=val-tol,
        upper_bound=val+tol,
        error_on_failure=True,
        execution_modes='test'))

    # Construct model
    mini_batch_size = 5
    num_epochs = 0
    return lbann.Model(mini_batch_size,
                       num_epochs,
                       layers=layers,
                       metrics=[metric],
                       callbacks=callbacks)

def construct_data_reader(lbann):
    """Construct Protobuf message for Python data reader.

    The Python data reader will import the current Python file to
    access the sample access functions.

    Args:
        lbann (module): Module for LBANN Python frontend

    """
    module_name = os.path.splitext(os.path.basename(current_file))[0]

    # Base data reader message
    message = lbann.reader_pb2.DataReader()

    # Training set data reader
    # TODO: This can be removed once
    # https://github.com/LLNL/lbann/issues/1098 is resolved.
    data_reader = message.reader.add()
    data_reader.name = 'python'
    data_reader.role = 'train'
    data_reader.percent_of_data_to_use = 1.0
    data_reader.python.module = module_name
    data_reader.python.module_dir = current_dir
    data_reader.python.sample_function = 'get_sample'
    data_reader.python.num_samples_function = 'num_samples'
    data_reader.python.sample_dims_function = 'sample_dims'

    # Test set data reader
    data_reader = message.reader.add()
    data_reader.name = 'python'
    data_reader.role = 'test'
    data_reader.percent_of_data_to_use = 1.0
    data_reader.python.module = module_name
    data_reader.python.module_dir = current_dir
    data_reader.python.sample_function = 'get_sample'
    data_reader.python.num_samples_function = 'num_samples'
    data_reader.python.sample_dims_function = 'sample_dims'

    return message

# ==============================================
# Setup PyTest
# ==============================================

# Generate test name based on file name
_test_name = os.path.splitext(os.path.basename(current_file))[0]

# Primary test method
def _test(cluster, executables, dir_name, compiler_name):
    tools.process_executable(_test_name, compiler_name, executables)

    # Import LBANN Python frontend
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
        'account': 'guests',
        'nodes': 1,
        'partition': 'pbatch'
    }
    experiment_dir = '{d}/bamboo/unit_tests/experiments/{t}_{c}'.format(
        d=dir_name, t=_test_name, c=compiler_name)
    error_file_name = '{e}/err.log'.format(
        e=experiment_dir, c=compiler_name)
    return_code = lbann.contrib.lc.launcher.run(
        trainer=trainer,
        model=model,
        data_reader=data_reader,
        optimizer=optimizer,
        experiment_dir=experiment_dir,
        job_name='lbann_{}'.format(_test_name),
        **kwargs)
    tools.assert_success(return_code, error_file_name)

# Construct methods that will be detected by PyTest
def _test_clang6(cluster, exes, dirname):
    _test(cluster, exes, dirname, 'clang6')
def _test_gcc7(cluster, exes, dirname):
    _test(cluster, exes, dirname, 'gcc7')
def _test_intel19(cluster, exes, dirname):
    _test(cluster, exes, dirname, 'intel19')
def _test_exe(cluster, dirname, exe):
    if exe is None:
        e = 'test_{}_exe: Non-local testing'.format(_test_name)
        print('Skip - ' + e)
        pytest.skip(e)
    exes = {'exe': exe}
    _test(cluster, exes, dirname, 'exe')
globals()['{}_clang6'.format(_test_name)] = _test_clang6
globals()['{}_gcc7'.format(_test_name)] = _test_gcc7
globals()['{}_intel19'.format(_test_name)] = _test_intel19
globals()['{}_exe'.format(_test_name)] = _test_exe
