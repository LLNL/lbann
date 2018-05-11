import sys
sys.path.insert(0, '../common_python')
import tools
import pytest
import os, sys

def test_unit_lbann2_reload(cluster, exes, dirname):
    lbann2  = exes['gcc4'] + '2'
    model_path = '{../../model_zoo/models/lenet_mnist/model_lenet_mnist.prototext,../../model_zoo/tests/model_lenet_mnist_lbann2ckpt.prototext}'
    command = tools.get_command(
        cluster=cluster, executable=lbann2, data_reader_name='mnist',
        data_filedir_default='/p/lscratchf/brainusr/datasets/MNIST',
        dir_name=dirname, 
        model_path=model_path,
        optimizer_name='sgd',
        num_epochs=2)
    return_code = os.system(command)
    if return_code != 0:
        sys.stderr.write('LBANN2 LeNet execution failed, exiting with error')
        sys.exit(1)

    
    os.system('mv lbann2_ckpt lbann2_nockpt')

    command = tools.get_command(
        cluster=cluster, executable=exe, num_nodes=1, num_processes=1,
        dir_name=dirname,
        data_filedir_default='/p/lscratchf/brainusr/datasets/MNIST',
        data_reader_name='mnist', model_folder='tests',
        model_name='lenet_mnist_ckpt', num_epochs=2, optimizer_name='sgd')   
    return_code_ckpt_1 = os.system(command)
    if return_code_ckpt_1 != 0:
        sys.stderr.write('LeNet (checkpoint) execution failed, exiting with error')
        sys.exit(1)
    
    command = tools.get_command(
        cluster=cluster, executable=lbann2, num_nodes=1, num_processes=1,
        dir_name=dirname,
        data_filedir_default='/p/lscratchf/brainusr/datasets/MNIST',
        data_reader_name='mnist',
        model_path='../../model_zoo/tests/model_lenet_mnist_lbann2ckpt.prototext', num_epochs=2, optimizer_name='sgd', ckpt_dir='ckpt/')
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
