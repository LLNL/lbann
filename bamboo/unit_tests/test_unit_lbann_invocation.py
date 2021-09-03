import sys
sys.path.insert(0, '../common_python')
import tools
import os
import shutil

def get_default_parameters(dir_name, two_models=True):
    data_reader_path = '{d}/model_zoo/data_readers/data_reader_mnist.prototext'.format(
        d=dir_name)
    model_path = '{d}/model_zoo/tests/model_lenet_mnist_ckpt.prototext'.format(
        d=dir_name)
    if two_models:
        model_path = '{{{mp},{mp}}}'.format(mp=model_path)
    optimizer_path = '{d}/model_zoo/optimizers/opt_sgd.prototext'.format(
        d=dir_name)
    return data_reader_path, model_path, optimizer_path


def get_file_names(dir_name, test_name):
    output_file_name = '{d}/bamboo/unit_tests/output/lbann_invocation_{t}_output.txt'.format(
        d=dir_name, t=test_name)
    error_file_name = '{d}/bamboo/unit_tests/error/lbann_invocation_{t}_error.txt'.format(
        d=dir_name, t=test_name)
    return output_file_name, error_file_name


# Run with python3 -m pytest -s test_unit_lbann_invocation.py -k 'test_unit_no_params_bad'
def test_unit_no_params_bad(cluster, dirname):
    print('TESTING: run lbann with no params; lbann should throw exception\n')
    (output_file_name, error_file_name) = get_file_names(dirname, 'no_params_bad')
    command = tools.get_command(
        cluster=cluster,
        exit_after_setup=True,
        num_processes=1,
        output_file_name=output_file_name,
        error_file_name=error_file_name
    )
    return_code = os.system(command)
    tools.assert_failure(return_code,
                         'Failed to load any prototext files',
                         error_file_name)


# Run with python3 -m pytest -s test_unit_lbann_invocation.py -k 'test_unit_one_model_bad'
def test_unit_one_model_bad(cluster, dirname):
    print('TESTING: run lbann with no optimizer or reader; lbann should throw exception\n')
    (_, model_path, _) = get_default_parameters(dirname, two_models=False)
    (output_file_name, error_file_name) = get_file_names(dirname, 'one_model_bad')
    command = tools.get_command(
        cluster=cluster,
        exit_after_setup=True,
        model_path=model_path,
        num_processes=1,
        output_file_name=output_file_name,
        error_file_name=error_file_name
    )
    return_code = os.system(command)
    tools.assert_failure(return_code,
                         'you specified 1 model filenames, and 0 optimizer filenames; you must specify 1 optimizer filenames',
                         error_file_name)


# Run with python3 -m pytest -s test_unit_lbann_invocation.py -k 'test_unit_two_models'
def test_unit_two_models(cluster, dirname):
    print('TESTING: run lbann with two models; lbann should throw exception\n')
    (data_reader_path, model_path, optimizer_path) = get_default_parameters(dirname)
    (output_file_name, error_file_name) = get_file_names(dirname, 'two_models')
    command = tools.get_command(
        cluster=cluster, data_reader_path=data_reader_path,
        data_filedir_default='/p/lscratchh/brainusr/datasets/MNIST',
        exit_after_setup=True,
        model_path=model_path,
        optimizer_path=optimizer_path,
        num_processes=1,
        output_file_name=output_file_name,
        error_file_name=error_file_name
    )
    return_code = os.system(command)
    tools.assert_failure(return_code,
                         'Arguments could not be parsed.',
                         error_file_name)


# Run with python3 -m pytest -s test_unit_lbann_invocation.py -k 'test_unit_missing_optimizer'
def test_unit_missing_optimizer(cluster, dirname):
    print('TESTING: run lbann with model, reader, but no optimizer; lbann should throw exception\n')
    (data_reader_path, model_path, _) = get_default_parameters(dirname, two_models=False)
    (output_file_name, error_file_name) = get_file_names(dirname, 'missing_optimizer')
    command = tools.get_command(
        cluster=cluster,
        data_reader_path=data_reader_path,
        data_filedir_default='/p/lscratchh/brainusr/datasets/MNIST',
        exit_after_setup=True, model_path=model_path,
        num_processes=1,
        output_file_name=output_file_name,
        error_file_name=error_file_name
    )
    return_code = os.system(command)
    tools.assert_failure(return_code,
                         'you specified 1 model filenames, and 0 optimizer filenames; you must specify 1 optimizer filenames',
                         error_file_name)


# Run with python3 -m pytest -s test_unit_lbann_invocation.py -k 'test_unit_missing_reader'
def test_unit_missing_reader(cluster, dirname):
    print('TESTING: run lbann with model, optimizer, but no reader; lbann should throw exception\n')
    (_, model_path, optimizer_path) = get_default_parameters(dirname, two_models=False)
    (output_file_name, error_file_name) = get_file_names(dirname, 'missing_reader')
    command = tools.get_command(
        cluster=cluster,
        exit_after_setup=True,
        model_path=model_path, optimizer_path=optimizer_path,
        num_processes=1,
        output_file_name=output_file_name,
        error_file_name=error_file_name
    )
    return_code = os.system(command)
    tools.assert_failure(return_code,
                         'you specified 1 model filenames, and 0 reader filenames; you must specify 1 reader filenames',
                         error_file_name)


# Run with python3 -m pytest -s test_unit_lbann_invocation.py -k 'test_unit_bad_params'
def test_unit_bad_params(cluster, dirname):
    exe = shutil.which('lbann')
    print('TESTING: run lbann with ill-formed param (exit_after_setup should have `--` not `-`) lbann should throw exception\n')
    (data_reader_path, model_path, optimizer_path) = get_default_parameters(
        dirname, two_models=False)
    (command_allocate, command_run, _, _) = tools.get_command(
        cluster=cluster,
        num_processes=1,
        return_tuple=True)
    (output_file_name, error_file_name) = get_file_names(dirname, 'bad_params')
    command_string = '{ca}{cr} {e} -exit_after_setup --reader={d} --model={m} --optimizer={o} > {ofn} 2> {efn}'.format(
        ca=command_allocate, cr=command_run, e=exe,
        d=data_reader_path, m=model_path, o=optimizer_path,
        ofn=output_file_name, efn=error_file_name
    )
    return_code = os.system(command_string)
    tools.assert_failure(return_code,
                         "Arguments could not be parsed.",
                         error_file_name)


# Run with python3 -m pytest -s test_unit_lbann_invocation.py -k 'test_unit_should_work'
def test_unit_should_work(cluster, dirname):
    print('TESTING: run lbann with model, reader, and optimizer; lbann should NOT throw exception\n')
    (data_reader_path, model_path, optimizer_path) = get_default_parameters(
        dirname, two_models=False)
    (output_file_name, error_file_name) = get_file_names(dirname, 'should_work')
    command = tools.get_command(
        cluster=cluster, data_reader_path=data_reader_path,
        data_filedir_default='/p/lscratchh/brainusr/datasets/MNIST',
        exit_after_setup=True, model_path=model_path,
        optimizer_path=optimizer_path,
        num_processes=1,
        output_file_name=output_file_name,
        error_file_name=error_file_name)
    return_code = os.system(command)
    tools.assert_success(return_code, error_file_name)
