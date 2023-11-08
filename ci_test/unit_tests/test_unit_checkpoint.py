import sys
sys.path.insert(0, '../common_python')
import tools
import pytest
import os
from filecmp import dircmp

def skeleton_checkpoint_lenet_shared(cluster, dir_name,
                                     weekly, data_reader_fraction):
    # Handle data
    if data_reader_fraction is None:
        data_reader_fraction = 0.01
    # No checkpointing, printing weights to files.
    output_file_name = '%s/ci_test/unit_tests/output/checkpoint_lenet_shared_no_checkpoint_output.txt' % (dir_name)
    error_file_name  = '%s/ci_test/unit_tests/error/checkpoint_lenet_shared_no_checkpoint_error.txt' % (dir_name)
    os.system('rm -rf ckpt_lenet_shared && mkdir ckpt_lenet_shared')
    no_ckpt_dir = 'ckpt_lenet_shared/no_ckpt'
    command = tools.get_command(
        cluster=cluster, num_nodes=1, num_processes=2,
        dir_name=dir_name,
        data_filedir_default='/p/lscratchh/brainusr/datasets/MNIST',
        data_reader_name='mnist', data_reader_fraction=data_reader_fraction,
        ckpt_dir=no_ckpt_dir, model_folder='tests',
        model_name='lenet_mnist_ckpt', num_epochs=2, optimizer_name='sgd',
        output_file_name=output_file_name, error_file_name=error_file_name, weekly=weekly)
    return_code_nockpt = os.system(command)
    tools.assert_success(return_code_nockpt, error_file_name)

    # Run to checkpoint, printing weights to files.
    output_file_name = '%s/ci_test/unit_tests/output/checkpoint_lenet_shared_checkpoint_output.txt' % (dir_name)
    error_file_name  = '%s/ci_test/unit_tests/error/checkpoint_lenet_shared_checkpoint_error.txt' % (dir_name)
    ckpt_dir = 'ckpt_lenet_shared/ckpt'
    command = tools.get_command(
        cluster=cluster, num_nodes=1, num_processes=2,
        dir_name=dir_name,
        data_filedir_default='/p/lscratchh/brainusr/datasets/MNIST',
        data_reader_name='mnist', data_reader_fraction=data_reader_fraction,
        ckpt_dir=ckpt_dir, model_folder='tests',
        model_name='lenet_mnist_ckpt', num_epochs=1, optimizer_name='sgd',
        output_file_name=output_file_name, error_file_name=error_file_name, weekly=weekly)
    return_code_ckpt_1 = os.system(command)
    tools.assert_success(return_code_ckpt_1, error_file_name)

    # Pick up from checkpoint, printing weights to files.
    output_file_name = '%s/ci_test/unit_tests/output/checkpoint_lenet_shared_restart_output.txt' % (dir_name)
    error_file_name  = '%s/ci_test/unit_tests/error/checkpoint_lenet_shared_restart_error.txt' % (dir_name)
    command = tools.get_command(
        cluster=cluster, num_nodes=1, num_processes=2,
        dir_name=dir_name,
        data_filedir_default='/p/lscratchh/brainusr/datasets/MNIST',
        data_reader_name='mnist', data_reader_fraction=data_reader_fraction,
        ckpt_dir=ckpt_dir, model_folder='tests',
        model_name='lenet_mnist_ckpt', num_epochs=2, optimizer_name='sgd',
        output_file_name=output_file_name, error_file_name=error_file_name, weekly=weekly)
    return_code_ckpt_2 = os.system(command)
    tools.assert_success(return_code_ckpt_2, error_file_name)

    dcmp = dircmp(ckpt_dir, no_ckpt_dir)
    fail, diffs, warns = tools.print_diff_files(dcmp)
    for w in warns:
        print(w)

    if fail:
        print()
        for d in diffs:
            print(d)
        path_prefix = '{d}/ci_test/unit_tests'.format(d=dir_name)
        raise AssertionError(
            'Compare {ncd} and {cd} in {p}'.format(
                ncd=no_ckpt_dir, cd=ckpt_dir, p=path_prefix))


def skeleton_checkpoint_lenet_distributed(cluster, dir_name,
                                          weekly, data_reader_fraction):
    # Handle data
    if data_reader_fraction is None:
        data_reader_fraction = 0.01

    # No checkpointing, printing weights to files.
    output_file_name = '%s/ci_test/unit_tests/output/checkpoint_lenet_distributed_no_checkpoint_output.txt' % (dir_name)
    error_file_name  = '%s/ci_test/unit_tests/error/checkpoint_lenet_distributed_no_checkpoint_error.txt' % (dir_name)
    os.system('rm -rf ckpt_lenet_distributed && mkdir ckpt_lenet_distributed')
    no_ckpt_dir = 'ckpt_lenet_distributed/no_ckpt'
    command = tools.get_command(
        cluster=cluster, num_nodes=1, num_processes=2,
        dir_name=dir_name,
        data_filedir_default='/p/lscratchh/brainusr/datasets/MNIST',
        data_reader_name='mnist', data_reader_fraction=data_reader_fraction,
        ckpt_dir=no_ckpt_dir, model_folder='tests',
        model_name='lenet_mnist_dist_ckpt', num_epochs=2, optimizer_name='sgd',
        output_file_name=output_file_name, error_file_name=error_file_name, weekly=weekly)
    return_code_nockpt = os.system(command)
    tools.assert_success(return_code_nockpt, error_file_name)

    # Run to checkpoint, printing weights to files.
    output_file_name = '%s/ci_test/unit_tests/output/checkpoint_lenet_distributed_checkpoint_output.txt' % (dir_name)
    error_file_name  = '%s/ci_test/unit_tests/error/checkpoint_lenet_distributed_checkpoint_error.txt' % (dir_name)
    ckpt_dir = 'ckpt_lenet_distributed/ckpt'
    command = tools.get_command(
        cluster=cluster, num_nodes=1, num_processes=2,
        dir_name=dir_name,
        data_filedir_default='/p/lscratchh/brainusr/datasets/MNIST',
        data_reader_name='mnist', data_reader_fraction=data_reader_fraction,
        ckpt_dir=ckpt_dir, model_folder='tests',
        model_name='lenet_mnist_dist_ckpt', num_epochs=1, optimizer_name='sgd',
        output_file_name=output_file_name, error_file_name=error_file_name, weekly=weekly)
    return_code_ckpt_1 = os.system(command)
    tools.assert_success(return_code_ckpt_1, error_file_name)

    # Pick up from checkpoint, printing weights to files.
    output_file_name = '%s/ci_test/unit_tests/output/checkpoint_lenet_distributed_restart_output.txt' % (dir_name)
    error_file_name  = '%s/ci_test/unit_tests/error/checkpoint_lenet_distributed_restart_error.txt' % (dir_name)
    command = tools.get_command(
        cluster=cluster, num_nodes=1, num_processes=2,
        dir_name=dir_name,
        data_filedir_default='/p/lscratchh/brainusr/datasets/MNIST',
        data_reader_name='mnist', data_reader_fraction=data_reader_fraction,
        ckpt_dir=ckpt_dir, model_folder='tests',
        model_name='lenet_mnist_dist_ckpt', num_epochs=2, optimizer_name='sgd',
        output_file_name=output_file_name, error_file_name=error_file_name, weekly=weekly)
    return_code_ckpt_2 = os.system(command)
    tools.assert_success(return_code_ckpt_2, error_file_name)

    dcmp = dircmp(ckpt_dir, no_ckpt_dir)
    fail, diffs, warns = tools.print_diff_files(dcmp)
    for w in warns:
        print(w)

    if fail:
        print()
        for d in diffs:
            print(d)
        path_prefix = '{d}/ci_test/unit_tests'.format(d=dir_name)
        raise AssertionError(
            'Compare {ncd} and {cd} in {p}'.format(
                ncd=no_ckpt_dir, cd=ckpt_dir, p=path_prefix))


# Run with python3 -m pytest -s test_unit_checkpoint.py -k 'test_unit_checkpoint_lenet_shared'
def test_unit_checkpoint_lenet_shared(cluster, dirname,
                                      weekly, data_reader_fraction):
    skeleton_checkpoint_lenet_shared(cluster, dirname,
                                     weekly, data_reader_fraction)


# Run with python3 -m pytest -s test_unit_checkpoint.py -k 'test_unit_checkpoint_lenet_distributed'
def test_unit_checkpoint_lenet_distributed(cluster, dirname, weekly, data_reader_fraction):
    skeleton_checkpoint_lenet_distributed(cluster, dirname, weekly, data_reader_fraction)
