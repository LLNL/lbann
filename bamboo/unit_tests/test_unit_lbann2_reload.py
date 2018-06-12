import sys
sys.path.insert(0, '../common_python')
import tools
import pytest
import os, sys

def skeleton_lbann2_reload(cluster, executables, dir_name, compiler_name):
    if compiler_name not in executables:
      pytest.skip('default_exes[%s] does not exist' % compiler_name)

    lbann2  = executables[compiler_name] + '2'
    model_path = '{../../model_zoo/models/lenet_mnist/model_lenet_mnist.prototext,../../model_zoo/tests/model_lenet_mnist_lbann2ckpt.prototext}'
    output_file_name = '%s/bamboo/unit_tests/output/lbann2_no_checkpoint_%s_output.txt' % (dir_name, compiler_name)
    error_file_name  = '%s/bamboo/unit_tests/error/lbann2_no_checkpoint_%s_error.txt' % (dir_name, compiler_name)
    command = tools.get_command(
        cluster=cluster, executable=lbann2, num_nodes=1, num_processes=2,
        data_reader_name='mnist',
        data_filedir_default='/p/lscratchf/brainusr/datasets/MNIST',
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

    output_file_name = '%s/bamboo/unit_tests/output/lbann2_checkpoint_%s_output.txt' % (dir_name, compiler_name)
    error_file_name  = '%s/bamboo/unit_tests/error/lbann2_checkpoint_%s_error.txt' % (dir_name, compiler_name)
    command = tools.get_command(
        cluster=cluster, executable=lbann2, num_nodes=1, num_processes=2,
        dir_name=dir_name,
        data_filedir_default='/p/lscratchf/brainusr/datasets/MNIST',
        data_reader_name='mnist', model_folder='tests',
        model_name='lenet_mnist_ckpt', num_epochs=2, optimizer_name='sgd',
        output_file_name=output_file_name,
        error_file_name=error_file_name)
    return_code_ckpt_1 = os.system(command)
    if return_code_ckpt_1 != 0:
        sys.stderr.write('LeNet (checkpoint) execution failed, exiting with error')
        sys.exit(1)

    output_file_name = '%s/bamboo/unit_tests/output/lbann2_restart_%s_output.txt' % (dir_name, compiler_name)
    error_file_name  = '%s/bamboo/unit_tests/error/lbann2_restart_%s_error.txt' % (dir_name, compiler_name)
    os.mkdir('lbann2_ckpt')
    command = tools.get_command(
        cluster=cluster, executable=lbann2, num_nodes=1, num_processes=2,
        dir_name=dir_name,
        data_filedir_default='/p/lscratchf/brainusr/datasets/MNIST',
        data_reader_name='mnist',
        model_path='../../model_zoo/tests/model_lenet_mnist_lbann2ckpt.prototext',
        num_epochs=2, optimizer_name='sgd', ckpt_dir='ckpt/',
        output_file_name=output_file_name,
        error_file_name=error_file_name)
    return_code_ckpt_2 = os.system(command)
    if return_code_ckpt_2 != 0:
        sys.stderr.write('LBANN2 LeNet weight reload failed, exiting with error')
        sys.exit(1)
    os.system('rm lbann2_ckpt/model0-epoch*')
    os.system('rm lbann2_nockpt/model0-epoch*')
    diff_test = os.system('diff -rq lbann2_ckpt/ lbann2_nockpt/')
    os.system('rm -rf ckpt')
    os.system('rm -rf lbann2_*')
    assert diff_test == 0

def test_unit_lbann2_reload_clang4(cluster, exes, dirname):
    if cluster in ['quartz']:
        pytest.skip('FIXME')
    skeleton_lbann2_reload(cluster, exes, dirname, 'clang4')

def test_unit_lbann2_reload_gcc4(cluster, exes, dirname):
  if cluster in ['surface']:
    pytest.skip('FIXME')
    # Surface Errors:
    # SystemExit: 1
    # [surface64:mpi_rank_0][error_sighandler] Caught error: Segmentation fault (signal 11)
    # srun: error: surface64: task 0: Segmentation fault (core dumped)
  skeleton_lbann2_reload(cluster, exes, dirname, 'gcc4')

def test_unit_lbann2_reload_gcc7(cluster, exes, dirname):
    if cluster in ['quartz']:
        pytest.skip('FIXME')
    skeleton_lbann2_reload(cluster, exes, dirname, 'gcc7')

def test_unit_lbann2_reload_intel18(cluster, exes, dirname):
    if cluster in ['quartz']:
        pytest.skip('FIXME')
    skeleton_lbann2_reload(cluster, exes, dirname, 'intel18')

# Run with python -m pytest -s test_unit_lbann2_reload.py -k 'test_unit_lbann2_reload_exe' --exe=<executable>
def test_unit_lbann2_reload_exe(cluster, dirname, exe):
    if exe == None:
        pytest.skip('Non-local testing')
    exes = {'exe' : exe}
    skeleton_lbann2_reload(cluster, exes, dirname, 'exe')
