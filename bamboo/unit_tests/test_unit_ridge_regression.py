import sys
sys.path.insert(0, '../common_python')
import tools
import pytest
import os

def skeleton_gradient_check(cluster, executables, dir_name, compiler_name):
    if compiler_name not in executables:
      pytest.skip('default_exes[%s] does not exist' % compiler_name)
    output_file_name = '%s/bamboo/unit_tests/output/gradient_check_ridge_regression_%s_output.txt' % (dir_name, compiler_name)
    error_file_name  = '%s/bamboo/unit_tests/error/gradient_check_ridge_regression_%s_error.txt' % (dir_name, compiler_name)
    command = tools.get_command(
        cluster=cluster, executable=executables[compiler_name], num_nodes=1, num_processes=1, dir_name=dir_name,
        data_filedir_default='/p/lscratchf/brainusr/datasets/MNIST', data_reader_name='mnist',
        model_folder='tests', model_name='mnist_ridge_regression', optimizer_name='adam',
        output_file_name=output_file_name, error_file_name=error_file_name)
    return_code = os.system(command)
    assert return_code == 0

def test_unit_gradient_check_clang4(cluster, exes, dirname):
    skeleton_gradient_check(cluster, exes, dirname, 'clang4')

def test_unit_gradient_check_gcc4(cluster, exes, dirname):
    skeleton_gradient_check(cluster, exes, dirname, 'gcc4')

def test_unit_gradient_check_gcc7(cluster, exes, dirname):
    skeleton_gradient_check(cluster, exes, dirname, 'gcc7')

def test_unit_gradient_check_intel18(cluster, exes, dirname):
    skeleton_gradient_check(cluster, exes, dirname, 'intel18')
