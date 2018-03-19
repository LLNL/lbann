import sys
sys.path.insert(0, '../common_python')
import tools
import pytest
import os

def test_unit_gradient_check_resnet(cluster, exe, dirname):
    command = tools.get_command(
        cluster=cluster, executable=exe, num_nodes=1, num_processes=1,
        dir_name=dirname, data_filedir_ray='/p/gscratchr/brainusr/datasets/MNIST',
        data_reader_name='mnist', model_folder='tests', model_name='mnist_resnet',
        optimizer_name='adam')
    return_code = os.system(command)
    assert return_code == 0
