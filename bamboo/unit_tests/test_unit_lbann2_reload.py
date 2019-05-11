import sys
sys.path.insert(0, '../common_python')
import tools
import pytest
import os, sys


def skeleton_lbann2_reload(cluster, executables, dir_name, compiler_name):
    if compiler_name not in executables:
      e = 'skeleton_lbann2_reload: default_exes[%s] does not exist' % compiler_name
      print('Skip - ' + e)
      pytest.skip(e)
    lbann2 = executables[compiler_name] + '2'

    # Delete directories / files if they happen to be around from the
    # previous build.
    os.system('rm -rf ckpt')
    os.system('rm -rf lbann2_*')


    # No checkpointing, printing weights to files.
    model_path = '{../../model_zoo/models/lenet_mnist/model_lenet_mnist.prototext,../../model_zoo/tests/model_lenet_mnist_lbann2ckpt.prototext}'
    output_file_name = '%s/bamboo/unit_tests/output/lbann2_no_checkpoint_%s_output.txt' % (dir_name, compiler_name)
    error_file_name  = '%s/bamboo/unit_tests/error/lbann2_no_checkpoint_%s_error.txt' % (dir_name, compiler_name)
    command = tools.get_command(
        cluster=cluster, executable=lbann2, num_nodes=1, num_processes=2,
        data_reader_name='mnist',
        data_filedir_default='/p/lscratchh/brainusr/datasets/MNIST',
        dir_name=dir_name,
        model_path=model_path,
        optimizer_name='sgd',
        num_epochs=2,
        output_file_name=output_file_name,
        error_file_name=error_file_name)

    os.mkdir('lbann2_ckpt')
    return_code = os.system(command)
    if return_code != 0:
        sys.stderr.write('LBANN2 LeNet execution failed, exiting with error')
        sys.exit(1)

    os.system('mv lbann2_ckpt lbann2_nockpt')

    # Run to checkpoint, printing weights to files.
    output_file_name = '%s/bamboo/unit_tests/output/lbann2_checkpoint_%s_output.txt' % (dir_name, compiler_name)
    error_file_name  = '%s/bamboo/unit_tests/error/lbann2_checkpoint_%s_error.txt' % (dir_name, compiler_name)
    command = tools.get_command(
        cluster=cluster, executable=lbann2, num_nodes=1, num_processes=2,
        dir_name=dir_name,
        data_filedir_default='/p/lscratchh/brainusr/datasets/MNIST',
        data_reader_name='mnist', model_folder='tests',
        model_name='lenet_mnist_ckpt', num_epochs=2, optimizer_name='sgd',
        output_file_name=output_file_name,
        error_file_name=error_file_name)
    return_code_ckpt_1 = os.system(command)
    if return_code_ckpt_1 != 0:
        sys.stderr.write(
            'LeNet (checkpoint) execution failed, exiting with error')
        sys.exit(1)

    # Pick up from checkpoint, printing weights to files.
    output_file_name = '%s/bamboo/unit_tests/output/lbann2_restart_%s_output.txt' % (dir_name, compiler_name)
    error_file_name  = '%s/bamboo/unit_tests/error/lbann2_restart_%s_error.txt' % (dir_name, compiler_name)
    os.mkdir('lbann2_ckpt')
    command = tools.get_command(
        cluster=cluster, executable=lbann2, num_nodes=1, num_processes=2,
        dir_name=dir_name,
        data_filedir_default='/p/lscratchh/brainusr/datasets/MNIST',
        data_reader_name='mnist',
        model_path='../../model_zoo/tests/model_lenet_mnist_lbann2ckpt.prototext',
        num_epochs=2, optimizer_name='sgd', ckpt_dir='ckpt/',
        output_file_name=output_file_name,
        error_file_name=error_file_name)
    return_code_ckpt_2 = os.system(command)
    if return_code_ckpt_2 != 0:
        sys.stderr.write(
            'LBANN2 LeNet weight reload failed, exiting with error')
        sys.exit(1)
    os.system('rm lbann2_ckpt/model0-epoch*')
    os.system('rm lbann2_nockpt/model0-epoch*')

    diff_result = os.system('diff -rq lbann2_ckpt/ lbann2_nockpt/')
    allow_epsilon_diff = False
    if allow_epsilon_diff and (diff_result != 0):
        equal_within_epsilon = True
        ckpt_files = os.listdir('lbann2_ckpt')
        for file_name in ckpt_files:
            ckpt_file = open('lbann2_ckpt/' + file_name, 'r')
            no_ckpt_file = open('lbann2_nockpt/' + file_name, 'r')
            for ckpt_line in ckpt_file:
                no_ckpt_line = next(no_ckpt_file)
                if ckpt_line != no_ckpt_line:
                    error_string = ('ckpt_line={ckpt_line},'
                                    ' nockpt_line={no_ckpt_line}').format(
                        ckpt_line=ckpt_line, no_ckpt_line=no_ckpt_line)
                    try:
                        ckpt_values = list(map(float, ckpt_line.split()))
                        no_ckpt_values = list(map(float, no_ckpt_line.split()))
                        num = len(ckpt_values)
                        if len(no_ckpt_values) == num:
                            for i in range(num):
                                if abs(ckpt_values[i] - no_ckpt_values[i]) > 0.5:
                                    # Not equal within epsilon.
                                    equal_within_epsilon = False
                                    print(error_string)
                        else:
                            # Length of lists don't match.
                            equal_within_epsilon = False
                            print(error_string)
                    except ValueError:
                        # Non-numerical diff.
                        equal_within_epsilon = False
                        print(error_string)
        if equal_within_epsilon:
            diff_result = 0
    os.system('rm -rf ckpt')
    os.system('rm -rf lbann2_*')
    assert diff_result == 0


def test_unit_lbann2_reload_clang4(cluster, exes, dirname):
    if cluster == 'catalyst':  # STILL ERRORS
        pytest.skip('FIXME')
    skeleton_lbann2_reload(cluster, exes, dirname, 'clang4')


def test_unit_lbann2_reload_gcc4(cluster, exes, dirname):
  skeleton_lbann2_reload(cluster, exes, dirname, 'gcc4')


def test_unit_lbann2_reload_gcc7(cluster, exes, dirname):
    if cluster in ['catalyst', 'pascal']:  # STILL ERRORS
        pytest.skip('FIXME')
    skeleton_lbann2_reload(cluster, exes, dirname, 'gcc7')


def test_unit_lbann2_reload_intel18(cluster, exes, dirname):
    skeleton_lbann2_reload(cluster, exes, dirname, 'intel18')


# Run with python -m pytest -s test_unit_lbann2_reload.py -k 'test_unit_lbann2_reload_exe' --exe=<executable>
def test_unit_lbann2_reload_exe(cluster, dirname, exe):
    if exe is None:
        e = 'test_unit_lbann2_reload_exe: Non-local testing'
        print('Skip - ' + e)
        pytest.skip(e)
    exes = {'exe': exe}
    skeleton_lbann2_reload(cluster, exes, dirname, 'exe')
