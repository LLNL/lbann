import sys
sys.path.insert(0, '../common_python')
import tools
import pytest
import os


def skeleton_checkpoint_lenet_shared(cluster, executables, dir_name,
                                     compiler_name, weekly, data_reader_percent):
    if compiler_name not in executables:
        e = 'skeleton_checkpoint_lenet_shared: default_exes[%s] does not exist' % compiler_name
        print('Skip - ' + e)
        pytest.skip(e)
    exe = executables[compiler_name]
    # Handle data
    if data_reader_percent is None:
        data_reader_percent = 0.01
    # No checkpointing, printing weights to files.
    output_file_name = '%s/bamboo/unit_tests/output/checkpoint_lenet_shared_no_checkpoint_%s_output.txt' % (dir_name, compiler_name)
    error_file_name  = '%s/bamboo/unit_tests/error/checkpoint_lenet_shared_no_checkpoint_%s_error.txt' % (dir_name, compiler_name)
    os.system('rm -rf ckpt_lenet_shared && mkdir ckpt_lenet_shared')
    no_ckpt_dir = 'ckpt_lenet_shared/no_ckpt_{c}'.format(c=compiler_name)
    command = tools.get_command(
        cluster=cluster, executable=exe, num_nodes=1, num_processes=2,
        dir_name=dir_name,
        data_filedir_default='/p/lscratchh/brainusr/datasets/MNIST',
        data_reader_name='mnist', data_reader_percent=data_reader_percent,
        ckpt_dir=no_ckpt_dir, model_folder='tests',
        model_name='lenet_mnist_ckpt', num_epochs=2, optimizer_name='sgd',
        output_file_name=output_file_name, error_file_name=error_file_name, weekly=weekly)
    return_code_nockpt = os.system(command)
    tools.assert_success(return_code_nockpt, error_file_name)

    # Run to checkpoint, printing weights to files.
    output_file_name = '%s/bamboo/unit_tests/output/checkpoint_lenet_shared_checkpoint_%s_output.txt' % (dir_name, compiler_name)
    error_file_name  = '%s/bamboo/unit_tests/error/checkpoint_lenet_shared_checkpoint_%s_error.txt' % (dir_name, compiler_name)
    ckpt_dir = 'ckpt_lenet_shared/ckpt_{c}'.format(c=compiler_name)
    command = tools.get_command(
        cluster=cluster, executable=exe, num_nodes=1, num_processes=2,
        dir_name=dir_name,
        data_filedir_default='/p/lscratchh/brainusr/datasets/MNIST',
        data_reader_name='mnist', data_reader_percent=data_reader_percent,
        ckpt_dir=ckpt_dir, model_folder='tests',
        model_name='lenet_mnist_ckpt', num_epochs=1, optimizer_name='sgd',
        output_file_name=output_file_name, error_file_name=error_file_name, weekly=weekly)
    return_code_ckpt_1 = os.system(command)
    tools.assert_success(return_code_ckpt_1, error_file_name)

    # Pick up from checkpoint, printing weights to files.
    output_file_name = '%s/bamboo/unit_tests/output/checkpoint_lenet_shared_restart_%s_output.txt' % (dir_name, compiler_name)
    error_file_name  = '%s/bamboo/unit_tests/error/checkpoint_lenet_shared_restart_%s_error.txt' % (dir_name, compiler_name)
    command = tools.get_command(
        cluster=cluster, executable=exe, num_nodes=1, num_processes=2,
        dir_name=dir_name,
        data_filedir_default='/p/lscratchh/brainusr/datasets/MNIST',
        data_reader_name='mnist', data_reader_percent=data_reader_percent,
        ckpt_dir=ckpt_dir, model_folder='tests',
        model_name='lenet_mnist_ckpt', num_epochs=2, optimizer_name='sgd',
        output_file_name=output_file_name, error_file_name=error_file_name, weekly=weekly)
    return_code_ckpt_2 = os.system(command)
    tools.assert_success(return_code_ckpt_2, error_file_name)

    diff_test = os.system('diff -r {ckpt} {no_ckpt}'.format(
        ckpt=ckpt_dir, no_ckpt=no_ckpt_dir))
    path_prefix = '{d}/bamboo/unit_tests/'.format(d=dir_name)
    if diff_test !=0:
        raise AssertionError('diff_test={dt}\nCompare {ncd} and {cd} in {p}'.format(
            dt=diff_test, ncd=no_ckpt_dir, cd=ckpt_dir, p=path_prefix))


def skeleton_checkpoint_lenet_distributed(cluster, executables, dir_name,
                                          compiler_name,
                                          weekly, data_reader_percent):
    if compiler_name not in executables:
        e = 'skeleton_checkpoint_lenet_distributed: default_exes[%s] does not exist' % compiler_name
        print('Skip - ' + e)
        pytest.skip(e)
    exe = executables[compiler_name]
    # Handle data
    if data_reader_percent is None:
        data_reader_percent = 0.01

    # No checkpointing, printing weights to files.
    output_file_name = '%s/bamboo/unit_tests/output/checkpoint_lenet_distributed_no_checkpoint_%s_output.txt' % (dir_name, compiler_name)
    error_file_name  = '%s/bamboo/unit_tests/error/checkpoint_lenet_distributed_no_checkpoint_%s_error.txt' % (dir_name, compiler_name)
    os.system('rm -rf ckpt_lenet_distributed && mkdir ckpt_lenet_distributed')
    no_ckpt_dir = 'ckpt_lenet_distributed/no_ckpt_{c}'.format(c=compiler_name)
    command = tools.get_command(
        cluster=cluster, executable=exe, num_nodes=1, num_processes=2,
        dir_name=dir_name,
        data_filedir_default='/p/lscratchh/brainusr/datasets/MNIST',
        data_reader_name='mnist', data_reader_percent=data_reader_percent,
        ckpt_dir=no_ckpt_dir, model_folder='tests',
        model_name='lenet_mnist_dist_ckpt', num_epochs=2, optimizer_name='sgd',
        output_file_name=output_file_name, error_file_name=error_file_name, weekly=weekly)
    return_code_nockpt = os.system(command)
    tools.assert_success(return_code_nockpt, error_file_name)

    # Run to checkpoint, printing weights to files.
    output_file_name = '%s/bamboo/unit_tests/output/checkpoint_lenet_distributed_checkpoint_%s_output.txt' % (dir_name, compiler_name)
    error_file_name  = '%s/bamboo/unit_tests/error/checkpoint_lenet_distributed_checkpoint_%s_error.txt' % (dir_name, compiler_name)
    ckpt_dir = 'ckpt_lenet_distributed/ckpt_{c}'.format(c=compiler_name)
    command = tools.get_command(
        cluster=cluster, executable=exe, num_nodes=1, num_processes=2,
        dir_name=dir_name,
        data_filedir_default='/p/lscratchh/brainusr/datasets/MNIST',
        data_reader_name='mnist', data_reader_percent=data_reader_percent,
        ckpt_dir=ckpt_dir, model_folder='tests',
        model_name='lenet_mnist_dist_ckpt', num_epochs=1, optimizer_name='sgd',
        output_file_name=output_file_name, error_file_name=error_file_name, weekly=weekly)
    return_code_ckpt_1 = os.system(command)
    tools.assert_success(return_code_ckpt_1, error_file_name)

    # Pick up from checkpoint, printing weights to files.
    output_file_name = '%s/bamboo/unit_tests/output/checkpoint_lenet_distributed_restart_%s_output.txt' % (dir_name, compiler_name)
    error_file_name  = '%s/bamboo/unit_tests/error/checkpoint_lenet_distributed_restart_%s_error.txt' % (dir_name, compiler_name)
    command = tools.get_command(
        cluster=cluster, executable=exe, num_nodes=1, num_processes=2,
        dir_name=dir_name,
        data_filedir_default='/p/lscratchh/brainusr/datasets/MNIST',
        data_reader_name='mnist', data_reader_percent=data_reader_percent,
        ckpt_dir=ckpt_dir, model_folder='tests',
        model_name='lenet_mnist_dist_ckpt', num_epochs=2, optimizer_name='sgd',
        output_file_name=output_file_name, error_file_name=error_file_name, weekly=weekly)
    return_code_ckpt_2 = os.system(command)
    tools.assert_success(return_code_ckpt_2, error_file_name)

    diff_test = os.system('diff -r {ckpt} {no_ckpt}'.format(
        ckpt=ckpt_dir, no_ckpt=no_ckpt_dir))
    path_prefix = '{d}/bamboo/unit_tests'.format(d=dir_name)
    if diff_test != 0:
        raise AssertionError(
            'diff_test={dt}\nCompare {ncd} and {cd} in {p}'.format(
                dt=diff_test, ncd=no_ckpt_dir, cd=ckpt_dir, p=path_prefix))


def test_unit_checkpoint_lenet_shared_clang6(cluster, exes, dirname,
                                             weekly, data_reader_percent):
    skeleton_checkpoint_lenet_shared(cluster, exes, dirname, 'clang6',
                                     weekly, data_reader_percent)


def test_unit_checkpoint_lenet_distributed_clang6(cluster, exes, dirname,
                                                  weekly, data_reader_percent):
    skeleton_checkpoint_lenet_distributed(cluster, exes, dirname, 'clang6',
                                          weekly, data_reader_percent)


def test_unit_checkpoint_lenet_shared_gcc7(cluster, exes, dirname,
                                           weekly, data_reader_percent):
    skeleton_checkpoint_lenet_shared(cluster, exes, dirname, 'gcc7',
                                     weekly, data_reader_percent)


def test_unit_checkpoint_lenet_distributed_gcc7(cluster, exes, dirname,
                                                weekly, data_reader_percent):
    skeleton_checkpoint_lenet_distributed(cluster, exes, dirname, 'gcc7',
                                          weekly, data_reader_percent)


def test_unit_checkpoint_lenet_shared_intel19(cluster, exes, dirname,
                                              weekly, data_reader_percent):
    skeleton_checkpoint_lenet_shared(cluster, exes, dirname, 'intel19',
                                     weekly, data_reader_percent)


def test_unit_checkpoint_lenet_distributed_intel19(cluster, exes, dirname,
                                                   weekly, data_reader_percent):
    skeleton_checkpoint_lenet_distributed(cluster, exes, dirname, 'intel19',
                                          weekly, data_reader_percent)


# Run with python3 -m pytest -s test_unit_checkpoint.py -k 'test_unit_checkpoint_lenet_shared_exe' --exe=<executable>
def test_unit_checkpoint_lenet_shared_exe(cluster, dirname, exe,
                                          weekly, data_reader_percent):
    if exe is None:
        e = 'test_unit_checkpoint_lenet_exe: Non-local testing'
        print('Skip - ' + e)
        pytest.skip(e)
    exes = {'exe': exe}
    skeleton_checkpoint_lenet_shared(cluster, exes, dirname, 'exe',
                                     weekly, data_reader_percent)


# Run with python3 -m pytest -s test_unit_checkpoint.py -k 'test_unit_checkpoint_lenet_distributed_exe' --exe=<executable>
def test_unit_checkpoint_lenet_distributed_exe(cluster, dirname, exe, weekly, data_reader_percent):
    if exe is None:
        e = 'test_unit_checkpoint_lenet_exe: Non-local testing'
        print('Skip - ' + e)
        pytest.skip(e)
    exes = {'exe': exe}
    skeleton_checkpoint_lenet_distributed(cluster, exes, dirname, 'exe', weekly, data_reader_percent)
