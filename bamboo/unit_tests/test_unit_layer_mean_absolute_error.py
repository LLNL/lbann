import sys
sys.path.insert(0, '../common_python')
import tools
import pytest
import os


def skeleton_layer_mean_absolute_error(cluster, executables, dir_name,
                                       compiler_name, weekly, data_reader_percent):
    if compiler_name not in executables:
      e = 'skeleton_layer_mean_absolute_error: default_exes[%s] does not exist' % compiler_name
      print('Skip - ' + e)
      pytest.skip(e)
    output_file_name = '%s/bamboo/unit_tests/output/layer_mean_absolute_error_%s_output.txt' % (dir_name, compiler_name)
    error_file_name  = '%s/bamboo/unit_tests/error/layer_mean_absolute_error_%s_error.txt' % (dir_name, compiler_name)
    command = tools.get_command(
        cluster=cluster, executable=executables[compiler_name], num_nodes=1,
        time_limit=10,
        num_processes=2, dir_name=dir_name,
        data_reader_name='synthetic',
        data_reader_percent=data_reader_percent,
        model_folder='tests/layer_tests', model_name='mean_absolute_error',
        optimizer_name='sgd',
        output_file_name=output_file_name, error_file_name=error_file_name, weekly=weekly)
    return_code = os.system(command)
    tools.assert_success(return_code, error_file_name)


def test_unit_layer_mean_absolute_error_clang6(cluster, exes, dirname,
                                               weekly, data_reader_percent):
    skeleton_layer_mean_absolute_error(cluster, exes, dirname, 'clang6',
                                       weekly, data_reader_percent)


def test_unit_layer_mean_absolute_error_gcc7(cluster, exes, dirname,
                                             weekly, data_reader_percent):
    skeleton_layer_mean_absolute_error(cluster, exes, dirname, 'gcc7',
                                       weekly, data_reader_percent)


def test_unit_layer_mean_absolute_error_intel19(cluster, exes, dirname,
                                                weekly, data_reader_percent):
    skeleton_layer_mean_absolute_error(cluster, exes, dirname, 'intel19',
                                       weekly, data_reader_percent)


# Run with python3 -m pytest -s test_unit_ridge_regression.py -k 'test_unit_layer_mean_absolute_error_exe' --exe=<executable>
def test_unit_layer_mean_absolute_error_exe(cluster, dirname, exe,
                                            weekly, data_reader_percent):
    if exe is None:
        e = 'test_unit_layer_mean_absolute_error_exe: Non-local testing'
        print('Skip - ' + e)
        pytest.skip(e)
    exes = {'exe': exe}
    skeleton_layer_mean_absolute_error(cluster, exes, dirname, 'exe',
                                       weekly, data_reader_percent)
