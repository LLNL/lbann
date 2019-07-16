import sys
sys.path.insert(0, '../common_python')
import os
import pytest
import tools


def skeleton_jag_reconstruction_loss(cluster, executables, dir_name, compiler_name):
    if compiler_name not in executables:
      e = 'skeleton_jag_reconstruction_loss: default_exes[%s] does not exist' % compiler_name
      print('Skip - ' + e)
      pytest.skip(e)
    output_file_name = '%s/bamboo/unit_tests/output/jag_reconstruction_loss_%s_output.txt' % (dir_name, compiler_name)
    error_file_name  = '%s/bamboo/unit_tests/error/jag_reconstruction_loss_%s_error.txt' % (dir_name, compiler_name)
    command = tools.get_command(
        cluster=cluster,
        executable=executables[compiler_name],
        num_nodes=16,
        num_processes=32,
        dir_name=dir_name,
        data_filedir_default='/p/lscratchh/brainusr/datasets/MNIST',
        data_reader_name='jag',
        metadata='model_zoo/models/jag/wae_cycle_gan/jag_100M_metadata.prototext',
        model_folder='tests',
        model_name='jag_single_layer_ae',
        optimizer_name='adam',
        output_file_name=output_file_name,
        error_file_name=error_file_name)
    return_code = os.system(command)
    assert return_code == 0


def test_unit_jag_reconstruction_loss_clang6(cluster, exes, dirname):
    skeleton_jag_reconstruction_loss(cluster, exes, dirname, 'clang6')


def test_unit_jag_reconstruction_loss_gcc7(cluster, exes, dirname):
    skeleton_jag_reconstruction_loss(cluster, exes, dirname, 'gcc7')


def test_unit_jag_reconstruction_loss_intel19(cluster, exes, dirname):
    skeleton_jag_reconstruction_loss(cluster, exes, dirname, 'intel19')


# Run with python -m pytest -s test_unit_ridge_regression.py -k 'test_unit_jag_reconstruction_loss_exe' --exe=<executable>
def test_unit_jag_reconstruction_loss_exe(cluster, dirname, exe):
    if exe is None:
        e = 'test_unit_jag_reconstruction_loss_exe: Non-local testing'
        print('Skip - ' + e)
        pytest.skip(e)
    exes = {'exe': exe}
    skeleton_jag_reconstruction_loss(cluster, exes, dirname, 'exe')
