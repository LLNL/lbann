import sys
sys.path.insert(0, '../common_python')
import tools
import pytest
import common_code


def skeleton_mnist_debug(cluster, dir_name, executables, compiler_name, weekly,
                         debug_build, should_log=False,
                         data_reader_percent=None):
    # If weekly or debug_build are true, then run the test.
    if not (weekly or debug_build):
        e = 'skeleton_mnist_debug: Not doing weekly or debug_build testing'
        print('Skip - ' + e)
        pytest.skip(e)
    if compiler_name not in executables:
        e = 'skeleton_mnist_debug: default_exes[%s] does not exist' % compiler_name
        print('Skip - ' + e)
        pytest.skip(e)
    model_name = 'lenet_mnist'
    output_file_name = '%s/bamboo/integration_tests/output/%s_%s_output.txt' %(dir_name, model_name, compiler_name)
    error_file_name = '%s/bamboo/integration_tests/error/%s_%s_error.txt' %(dir_name, model_name, compiler_name)
    command = tools.get_command(
        cluster=cluster, executable=executables[compiler_name],
        # Allocation/Run Parameters
        num_nodes=1, partition='pbatch', time_limit=100,
        # LBANN Parameters
        dir_name=dir_name,
        data_filedir_default='/p/lscratchh/brainusr/datasets/MNIST',
        data_reader_name='mnist',
        data_reader_percent=data_reader_percent,
        model_folder='models/' + model_name,
        model_name=model_name, num_epochs=5, optimizer_name='adagrad',
        # Error/Output Redirect
        error_file_name=error_file_name,
        output_file_name=output_file_name,
        # Misc. Parameters
        weekly=weekly)
    common_code.run_lbann(command, model_name, output_file_name,
                          error_file_name, should_log)


def skeleton_cifar_debug(cluster, dir_name, executables, compiler_name, weekly,
                         debug_build, should_log=False,
                         data_reader_percent=None):
    # If weekly or debug_build are true, then run the test.
    if not (weekly or debug_build):
        e = 'skeleton_cifar_debug: Not doing weekly or debug_build testing'
        print('Skip - ' + e)
        pytest.skip(e)
    if cluster == 'ray':
        e = 'skeleton_cifar_debug: cifar not operational on Ray'
        print('Skip - ' + e)
        pytest.skip(e)
    if compiler_name not in executables:
        e = 'skeleton_cifar_debug: default_exes[%s] does not exist' % compiler_name
        print('Skip - ' + e)
        pytest.skip(e)
    model_name = 'autoencoder_cifar10'
    output_file_name = '%s/bamboo/integration_tests/output/%s_%s_output.txt' %(dir_name, model_name, compiler_name)
    error_file_name = '%s/bamboo/integration_tests/error/%s_%s_error.txt' %(dir_name, model_name, compiler_name)
    command = tools.get_command(
        cluster=cluster, executable=executables[compiler_name],
        # Allocation/Run Parameters
        num_nodes=1, partition='pbatch', time_limit=100,
        # LBANN Parameters
        dir_name=dir_name,
        data_filename_train_default='/p/lscratchh/brainusr/datasets/cifar10-bin/data_all.bin',
        data_filename_test_default='/p/lscratchh/brainusr/datasets/cifar10-bin/test_batch.bin',
        data_reader_name='cifar10', data_reader_percent=data_reader_percent,
        model_folder='models/' + model_name,
        model_name='conv_' + model_name, num_epochs=5, optimizer_name='adagrad',
        # Error/Output Redirect
        error_file_name=error_file_name,
        output_file_name=output_file_name,
        # Misc. Parameters
        weekly=weekly)
    common_code.run_lbann(command, model_name, output_file_name,
                          error_file_name, should_log)


def test_integration_mnist_clang6_debug(cluster, dirname, exes, weekly,
                                        debug_build, data_reader_percent):
    skeleton_mnist_debug(cluster, dirname, exes, 'clang6_debug', weekly,
                         debug_build, data_reader_percent)


def test_integration_cifar_clang6_debug(cluster, dirname, exes, weekly,
                                        debug_build, data_reader_percent):
    skeleton_cifar_debug(cluster, dirname, exes, 'clang6_debug', weekly,
                         debug_build, data_reader_percent)


def test_integration_mnist_gcc7_debug(cluster, dirname, exes, weekly,
                                      debug_build, data_reader_percent):
    skeleton_mnist_debug(cluster, dirname, exes, 'gcc7_debug', weekly,
                         debug_build, data_reader_percent)


def test_integration_cifar_gcc7_debug(cluster, dirname, exes, weekly,
                                      debug_build, data_reader_percent):
    skeleton_cifar_debug(cluster, dirname, exes, 'gcc7_debug', weekly,
                         debug_build, data_reader_percent)


def test_integration_mnist_intel19_debug(cluster, dirname, exes, weekly,
                                         debug_build, data_reader_percent):
    skeleton_mnist_debug(cluster, dirname, exes, 'intel19_debug', weekly,
                         debug_build, data_reader_percent)


def test_integration_cifar_intel19_debug(cluster, dirname, exes, weekly,
                                         debug_build, data_reader_percent):
    skeleton_cifar_debug(cluster, dirname, exes, 'intel19_debug', weekly,
                         debug_build, data_reader_percent)


# Run with python3 -m pytest -s test_integration_debug.py -k 'test_integration_mnist_exe' --exe=<executable>
def test_integration_mnist_exe(cluster, dirname, exe, weekly,
                               data_reader_percent):
    if exe is None:
        e = 'test_integration_mnist_exe: Non-local testing'
        print('Skip - ' + e)
        pytest.skip(e)
    exes = {'exe': exe}
    debug_build = True
    skeleton_mnist_debug(cluster, dirname, exes, 'exe', weekly, debug_build,
                         data_reader_percent=data_reader_percent)


# Run with python3 -m pytest -s test_integration_debug.py -k 'test_integration_cifar_exe' --exe=<executable>
def test_integration_cifar_exe(cluster, dirname, exe, weekly,
                               data_reader_percent):
    if exe == None:
        e = 'test_integration_cifar_exe: Non-local testing'
        print('Skip - ' + e)
        pytest.skip(e)
    exes = {'exe': exe}
    debug_build=True
    skeleton_cifar_debug(cluster, dirname, exes, 'exe', weekly, debug_build,
                         data_reader_percent=data_reader_percent)
