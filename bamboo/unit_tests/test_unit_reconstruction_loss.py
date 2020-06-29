import sys
sys.path.insert(0, '../common_python')
import os
import pytest
import tools


def skeleton_jag_reconstruction_loss(cluster, executables, dir_name, compiler_name,
                                     weekly, data_reader_percent):
    if compiler_name not in executables:
        e = 'skeleton_jag_reconstruction_loss: default_exes[%s] does not exist' % compiler_name
        print('Skip - ' + e)
        pytest.skip(e)
    if cluster == 'ray':
        e = 'skeleton_jag_reconstruction_loss: dataset does not exist on %s' % cluster
        print('Skip - ' + e)
        pytest.skip(e)
    #if cluster == 'lassen':
        #e = 'skeleton_jag_reconstruction_loss: FIXME dataset consistency issues on Lassen'
        #print('Skip - ' + e)
        #pytest.skip(e)
    output_file_name = '%s/bamboo/unit_tests/output/jag_reconstruction_loss_%s_output.txt' % (dir_name, compiler_name)
    error_file_name  = '%s/bamboo/unit_tests/error/jag_reconstruction_loss_%s_error.txt' % (dir_name, compiler_name)
    command = tools.get_command(
        cluster=cluster,
        executable=executables[compiler_name],
        num_nodes=2,
        num_processes=32,
        disable_cuda=1,
        dir_name=dir_name,
        data_filedir_train_default='/p/lscratchh/brainusr/datasets/10MJAG/1M_A/100K4trainers',
        data_filedir_test_default='/p/lscratchh/brainusr/datasets/10MJAG/1M_A/100K16trainers',
        data_reader_name='jag',
        data_reader_percent='prototext',
        metadata='applications/physics/data/jag_100M_metadata.prototext',
        model_folder='tests',
        model_name='jag_single_layer_ae',
        optimizer_name='adam',
        output_file_name=output_file_name,
        error_file_name=error_file_name, weekly=weekly)
    return_code = os.system(command)
    tools.assert_success(return_code, error_file_name)


def test_unit_jag_reconstruction_loss_clang6(cluster, exes, dirname,
                                             weekly, data_reader_percent):
    skeleton_jag_reconstruction_loss(cluster, exes, dirname, 'clang6',
                                     weekly, data_reader_percent)


def test_unit_jag_reconstruction_loss_gcc7(cluster, exes, dirname,
                                           weekly, data_reader_percent):
    skeleton_jag_reconstruction_loss(cluster, exes, dirname, 'gcc7',
                                     weekly, data_reader_percent)


def test_unit_jag_reconstruction_loss_intel19(cluster, exes, dirname,
                                              weekly, data_reader_percent):
    skeleton_jag_reconstruction_loss(cluster, exes, dirname, 'intel19',
                                     weekly, data_reader_percent)


# Run with python3 -m pytest -s test_unit_ridge_regression.py -k 'test_unit_jag_reconstruction_loss_exe' --exe=<executable>
def test_unit_jag_reconstruction_loss_exe(cluster, dirname, exe,
                                          weekly, data_reader_percent):
    if exe is None:
        e = 'test_unit_jag_reconstruction_loss_exe: Non-local testing'
        print('Skip - ' + e)
        pytest.skip(e)
    exes = {'exe': exe}
    skeleton_jag_reconstruction_loss(cluster, exes, dirname, 'exe',
                                     weekly, data_reader_percent)
