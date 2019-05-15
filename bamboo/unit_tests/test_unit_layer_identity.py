import sys
sys.path.insert(0, '../common_python')
import tools
import pytest
import os


def skeleton_layer_identity(cluster, executables, dir_name, compiler_name):
    if compiler_name not in executables:
        e = 'skeleton_layer_identity: default_exes[%s] does not exist' % compiler_name
        print('Skip - ' + e)
        pytest.skip(e)
    output_file_name = '%s/bamboo/unit_tests/output/layer_identity_%s_output.txt' % (dir_name, compiler_name)
    error_file_name  = '%s/bamboo/unit_tests/error/layer_identity_%s_error.txt' % (dir_name, compiler_name)
    command = tools.get_command(
        cluster=cluster, executable=executables[compiler_name], num_nodes=1,
        num_processes=2, dir_name=dir_name,
        data_filedir_default='', data_reader_name='synthetic',
        model_folder='tests/layer_tests', model_name='identity',
        optimizer_name='sgd',
        output_file_name=output_file_name, error_file_name=error_file_name)
    return_code = os.system(command)
    assert return_code == 0


def test_unit_layer_identity_clang4(cluster, exes, dirname):
    skeleton_layer_identity(cluster, exes, dirname, 'clang4')


def test_unit_layer_identity_gcc4_check(cluster, exes, dirname):
    skeleton_layer_identity(cluster, exes, dirname, 'gcc4')


def test_unit_layer_identity_gcc7(cluster, exes, dirname):
    skeleton_layer_identity(cluster, exes, dirname, 'gcc7')


def test_unit_layer_identity_intel18(cluster, exes, dirname):
    skeleton_layer_identity(cluster, exes, dirname, 'intel18')


# Run with python -m pytest -s test_unit_layer_identity.py -k 'test_unit_layer_identity_exe' --exe=<executable>
def test_unit_layer_identity_exe(cluster, dirname, exe):
    if exe is None:
        e = 'test_unit_layer_identity_exe: Non-local testing'
        print('Skip - ' + e)
        pytest.skip(e)
    exes = {'exe': exe}
    skeleton_layer_identity(cluster, exes, dirname, 'exe')
