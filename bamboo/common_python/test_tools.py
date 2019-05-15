import pytest
import tools

# This test isn't in a directory to be run from Bamboo
# Run locally with python -m pytest -s

def test_command_catalyst():
    actual = tools.get_command(cluster='catalyst', executable='exe', num_nodes=20, partition='pdebug', time_limit=30, num_processes=40, dir_name='dir', data_filedir_default='lscratchh/filedir', data_reader_name='mnist', data_reader_percent=0.10, exit_after_setup=True, mini_batch_size=15, model_folder='models/folder', model_name='lenet', num_epochs=7, optimizer_name='adagrad', processes_per_model=10, output_file_name='output_file', error_file_name='error_file', check_executable_existence=False)
    expected = 'salloc --nodes=20 --partition=pdebug --time=30 srun --ntasks=40 exe --reader=dir/model_zoo/data_readers/data_reader_mnist.prototext --data_reader_percent=0.100000 --exit_after_setup --mini_batch_size=15 --model=dir/model_zoo/models/folder/model_lenet.prototext --num_epochs=7 --optimizer=dir/model_zoo/optimizers/opt_adagrad.prototext --procs_per_model=10 > output_file 2> error_file'
    assert actual == expected

def test_command_pascal():
    actual = tools.get_command(cluster='pascal', executable='exe', num_nodes=20, partition='pdebug', time_limit=30, num_processes=40, dir_name='dir', data_filedir_default='lscratchh/filedir', data_reader_name='mnist', data_reader_percent=0.10, exit_after_setup=True, mini_batch_size=15, model_folder='models/folder', model_name='lenet', num_epochs=7, optimizer_name='adagrad', processes_per_model=10, output_file_name='output_file', error_file_name='error_file', check_executable_existence=False)
    expected = 'salloc --nodes=20 --partition=pdebug --time=30 srun --ntasks=40 exe --reader=dir/model_zoo/data_readers/data_reader_mnist.prototext --data_reader_percent=0.100000 --exit_after_setup --mini_batch_size=15 --model=dir/model_zoo/models/folder/model_lenet.prototext --num_epochs=7 --optimizer=dir/model_zoo/optimizers/opt_adagrad.prototext --procs_per_model=10 > output_file 2> error_file'
    assert actual == expected

def test_command_quartz():
    actual = tools.get_command(cluster='quartz', executable='exe', num_nodes=20, partition='pdebug', time_limit=30, num_processes=40, dir_name='dir', data_filedir_default='lscratchh/filedir', data_reader_name='mnist', data_reader_percent=0.10, exit_after_setup=True, mini_batch_size=15, model_folder='models/folder', model_name='lenet', num_epochs=7, optimizer_name='adagrad', processes_per_model=10, output_file_name='output_file', error_file_name='error_file', check_executable_existence=False)
    expected = 'salloc --nodes=20 --partition=pdebug --time=30 srun --ntasks=40 exe --data_filedir=lscratchh/filedir --reader=dir/model_zoo/data_readers/data_reader_mnist.prototext --data_reader_percent=0.100000 --exit_after_setup --mini_batch_size=15 --model=dir/model_zoo/models/folder/model_lenet.prototext --num_epochs=7 --optimizer=dir/model_zoo/optimizers/opt_adagrad.prototext --procs_per_model=10 > output_file 2> error_file'
    assert actual == expected
    
def test_command_surface():
    actual = tools.get_command(cluster='surface', executable='exe', num_nodes=20, partition='pdebug', time_limit=30, num_processes=40, dir_name='dir', data_filedir_default='lscratchh/filedir', data_reader_name='mnist', data_reader_percent=0.10, exit_after_setup=True, mini_batch_size=15, model_folder='models/folder', model_name='lenet', num_epochs=7, optimizer_name='adagrad', processes_per_model=10, output_file_name='output_file', error_file_name='error_file', check_executable_existence=False)
    expected = 'salloc --nodes=20 --partition=pbatch --time=30 srun --ntasks=40 exe --reader=dir/model_zoo/data_readers/data_reader_mnist.prototext --data_reader_percent=0.100000 --exit_after_setup --mini_batch_size=15 --model=dir/model_zoo/models/folder/model_lenet.prototext --num_epochs=7 --optimizer=dir/model_zoo/optimizers/opt_adagrad.prototext --procs_per_model=10 > output_file 2> error_file'
    assert actual == expected

def test_command_ray():
    actual = tools.get_command(cluster='ray', executable='exe', num_nodes=20, partition='pdebug', time_limit=30, num_processes=40, dir_name='dir', data_filedir_default='lscratchh/filedir', data_reader_name='mnist', data_reader_percent=0.10, exit_after_setup=True, mini_batch_size=15, model_folder='models/folder', model_name='lenet', num_epochs=7, optimizer_name='adagrad', processes_per_model=10, output_file_name='output_file', error_file_name='error_file', check_executable_existence=False)
    expected = 'bsub -x -G guests -Is -n 40 -q pdebug -R "span[ptile=2]" -W 30 mpirun -np 40 -N 2 exe --data_filedir=gscratchr/filedir --reader=dir/model_zoo/data_readers/data_reader_mnist.prototext --data_reader_percent=0.100000 --exit_after_setup --mini_batch_size=15 --model=dir/model_zoo/models/folder/model_lenet.prototext --num_epochs=7 --optimizer=dir/model_zoo/optimizers/opt_adagrad.prototext --procs_per_model=10 > output_file 2> error_file'
    assert actual == expected

# Test error cases ############################################################

def test_blacklisted_substrings():
    try:
        tools.get_command('ray', 'exe', partition=';', optimizer_path='--model=new_model', check_executable_existence=False)
    except Exception as e:
        actual = str(e)
        expected = 'Invalid character(s): ; contains ; , --model=new_model contains --'
        assert actual == expected

def test_unsupported_cluster():
    try:
        tools.get_command('quartz', 'exe', check_executable_existence=False)
    except Exception as e:
        actual = str(e)
        expected = 'Unsupported Cluster: quartz'
        assert actual == expected

def test_bad_model_1():
    try:
        tools.get_command('ray', 'exe', dir_name='dir', model_folder='folder', model_name='name', model_path='path', check_executable_existence=False)
    except Exception as e:
        actual = str(e)
        expected = 'Invalid Usage: model_path is set but so is at least one of model folder and model_name'
        assert actual == expected

def test_bad_model_2():
    try:
        tools.get_command('ray', 'exe', dir_name='dir', model_folder='folder', model_path='path', check_executable_existence=False)
    except Exception as e:
        actual = str(e)
        expected = 'Invalid Usage: model_path is set but so is at least one of model folder and model_name'
        assert actual == expected

def test_bad_model_3():
    try:
        tools.get_command('ray', 'exe', dir_name='dir', model_name='name',  model_path='path', check_executable_existence=False)
    except Exception as e:
        actual = str(e)
        expected = 'Invalid Usage: model_path is set but so is at least one of model folder and model_name'
        assert actual == expected

def test_bad_model_4():
    try:
        tools.get_command('ray', 'exe', dir_name='dir', model_folder='folder', check_executable_existence=False)
    except Exception as e:
        actual = str(e)
        expected = 'Invalid Usage: model_folder set but not model_name.'
        assert actual == expected

def test_bad_model_5():
    try:
        tools.get_command('ray', 'exe', dir_name='dir', model_name='name', check_executable_existence=False)
    except Exception as e:
        actual = str(e)
        expected = 'Invalid Usage: model_name set but not model_folder.'
        assert actual == expected

def test_bad_data_reader():
    try:
        tools.get_command('catalyst', 'exe', dir_name='dir', data_reader_name='name', data_reader_path='path', check_executable_existence=False)
    except Exception as e:
        actual = str(e)
        expected = 'Invalid Usage: data_reader_path is set but so is data_reader_name , data_reader_name or data_reader_path is set but not data_filedir_default. If a data reader is provided, the default filedir must be set. This allows for determining what the filedir should be on each cluster. Alternatively, some or all of [data_filedir_train_default, data_filename_train_default, data_filedir_test_default, data_filename_test_default] can be set.'
        assert actual == expected

def test_bad_optimizer():
    try:
        tools.get_command('ray', 'exe', dir_name='dir', optimizer_name='name', optimizer_path='path', check_executable_existence=False)
    except Exception as e:
        actual = str(e)
        expected = 'Invalid Usage: optimizer_path is set but so is optimizer_name'
        assert actual == expected

def test_bad_dir_name_1():
    try:
        tools.get_command('ray', 'exe', dir_name='dir', check_executable_existence=False)
    except Exception as e:
	actual = str(e)
	expected = 'Invalid Usage: dir_name set but none of model_folder, model_name, data_reader_name, optimizer_name are.'
        assert actual == expected

def test_bad_dir_name_2():
    try:
        tools.get_command('ray', 'exe', model_folder='folder', check_executable_existence=False)
    except Exception as e:
        actual = str(e)
        expected = 'Invalid Usage: dir_name is not set but at least one of model_folder, model_name, data_reader_name, optimizer_name is.'
        assert actual == expected

def test_bad_dir_name_3():
    try:
        tools.get_command('ray', 'exe', model_name='name', check_executable_existence=False)
    except Exception as e:
	actual = str(e)
	expected = 'Invalid Usage: dir_name is not set but at least one of model_folder, model_name, data_reader_name, optimizer_name is.'
        assert actual == expected

def test_bad_dir_name_4():
    try:
        tools.get_command('catalyst', 'exe', data_reader_name='name', check_executable_existence=False)
    except Exception as e:
        actual = str(e)
        expected = 'Invalid Usage: dir_name is not set but at least one of model_folder, model_name, data_reader_name, optimizer_name is. , data_reader_name or data_reader_path is set but not data_filedir_default. If a data reader is provided, the default filedir must be set. This allows for determining what the filedir should be on each cluster. Alternatively, some or all of [data_filedir_train_default, data_filename_train_default, data_filedir_test_default, data_filename_test_default] can be set.'
        assert actual == expected

def test_bad_dir_name_5():
    try:
        tools.get_command('ray', 'exe', optimizer_name='name', check_executable_existence=False)
    except Exception as e:
	actual = str(e)
	expected = 'Invalid Usage: dir_name is not set but at least one of model_folder, model_name, data_reader_name, optimizer_name is.'
        assert actual == expected

def test_bad_data_filedir_1():
    try:
        tools.get_command('ray', 'exe', dir_name='dir', data_reader_name='name', data_filedir_default='filedir', data_filedir_train_default='a',
                          check_executable_existence=False)
    except Exception as e:
        actual = str(e)
        expected = 'Invalid Usage: data_fildir_default set but so is at least one of [data_filedir_train_default, data_filename_train_default, data_filedir_test_default, data_filename_test_default]'
        assert actual == expected

def test_bad_data_filedir_2():
    try:
        tools.get_command('ray', 'exe', dir_name='dir', data_reader_name='name', data_filedir_default='filedir', data_filename_train_default='b',
                          check_executable_existence=False)
    except Exception as e:
        actual = str(e)
        expected = 'Invalid Usage: data_fildir_default set but so is at least one of [data_filedir_train_default, data_filename_train_default, data_filedir_test_default, data_filename_test_default]'
        assert actual == expected


def test_bad_data_filedir_3():
    try:
        tools.get_command('ray', 'exe', dir_name='dir', data_reader_name='name', data_filedir_default='filedir', data_filedir_test_default='c',
                          check_executable_existence=False)
    except Exception as e:
        actual = str(e)
        expected = 'Invalid Usage: data_fildir_default set but so is at least one of [data_filedir_train_default, data_filename_train_default, data_filedir_test_default, data_filename_test_default]'
        assert actual == expected

def test_bad_data_filedir_4():
    try:
        tools.get_command('ray', 'exe', dir_name='dir', data_reader_name='name', data_filedir_default='filedir', data_filename_test_default='d',
                          check_executable_existence=False)
    except Exception as e:
        actual = str(e)
        expected = 'Invalid Usage: data_fildir_default set but so is at least one of [data_filedir_train_default, data_filename_train_default, data_filedir_test_default, data_filename_test_default]'
        assert actual == expected

def test_bad_data_filedir_5():
    try:
        tools.get_command('ray', 'exe', data_reader_path='path', data_filedir_default='filedir', data_filedir_train_default='e', check_executable_existence=False)
    except Exception as e:
        actual = str(e)
        expected = 'Invalid Usage: data_fildir_default set but so is at least one of [data_filedir_train_default, data_filename_train_default, data_filedir_test_default, data_filename_test_default]'
        assert actual == expected

def test_bad_data_filedir_6():
    try:
        tools.get_command('ray', 'exe', data_reader_path='path', data_filedir_default='filedir', data_filename_train_default='f', check_executable_existence=False)
    except Exception as e:
        actual = str(e)
        expected = 'Invalid Usage: data_fildir_default set but so is at least one of [data_filedir_train_default, data_filename_train_default, data_filedir_test_default, data_filename_test_default]'
        assert actual == expected


def test_bad_data_filedir_7():
    try:
        tools.get_command('ray', 'exe', data_reader_path='path', data_filedir_default='filedir', data_filedir_test_default='g', check_executable_existence=False)
    except Exception as e:
        actual = str(e)
        expected = 'Invalid Usage: data_fildir_default set but so is at least one of [data_filedir_train_default, data_filename_train_default, data_filedir_test_default, data_filename_test_default]'
        assert actual == expected

def test_bad_data_filedir_8():
    try:
        tools.get_command('ray', 'exe', data_reader_path='path', data_filedir_default='filedir', data_filename_test_default='h', check_executable_existence=False)
    except Exception as e:
        actual = str(e)
        expected = 'Invalid Usage: data_fildir_default set but so is at least one of [data_filedir_train_default, data_filename_train_default, data_filedir_test_default, data_filename_test_default]'
        assert actual == expected

def test_bad_data_filedir_9():
    try:
        tools.get_command('ray', 'exe', dir_name='dir', data_reader_name='name', check_executable_existence=False)
    except Exception as e:
        actual = str(e)
        expected = 'Invalid Usage: data_reader_name or data_reader_path is set but not data_filedir_default. If a data reader is provided, the default filedir must be set. This allows for determining what the filedir should be on each cluster. Alternatively, some or all of [data_filedir_train_default, data_filename_train_default, data_filedir_test_default, data_filename_test_default] can be set.'
        assert actual == expected

def test_bad_data_filedir_10():
    try:
        tools.get_command('ray', 'exe', data_reader_path='path', check_executable_existence=False)
    except Exception as e:
        actual = str(e)
        expected = 'Invalid Usage: data_reader_name or data_reader_path is set but not data_filedir_default. If a data reader is provided, the default filedir must be set. This allows for determining what the filedir should be on each cluster. Alternatively, some or all of [data_filedir_train_default, data_filename_train_default, data_filedir_test_default, data_filename_test_default] can be set.'
        assert actual == expected

def test_bad_data_filedir_11():
    try:
        tools.get_command('ray', 'exe', data_filedir_default='filedir', check_executable_existence=False)
    except Exception as e:
        actual = str(e)
        expected = 'Invalid Usage: data_filedir_default set but neither data_reader_name or data_reader_path are.'
	assert actual == expected

def test_bad_data_filedir_12():
    try:
        tools.get_command('ray', 'exe', data_filedir_train_default='a', check_executable_existence=False)
    except Exception as e:
        actual = str(e)
        expected = 'Invalid Usage: At least one of [data_filedir_train_default, data_filename_train_default, data_filedir_test_default, data_filename_test_default] is set, but neither data_reader_name or data_reader_path are.'
        assert actual == expected


def test_bad_data_filedir_13():
    try:
        tools.get_command('ray', 'exe', data_filename_train_default='b', check_executable_existence=False)
    except Exception as e:
        actual = str(e)
        expected = 'Invalid Usage: At least one of [data_filedir_train_default, data_filename_train_default, data_filedir_test_default, data_filename_test_default] is set, but neither data_reader_name or data_reader_path are.'
        assert actual == expected


def test_bad_data_filedir_14():
    try:
        tools.get_command('ray', 'exe', data_filedir_test_default='c', check_executable_existence=False)
    except Exception as e:
        actual = str(e)
        expected = 'Invalid Usage: At least one of [data_filedir_train_default, data_filename_train_default, data_filedir_test_default, data_filename_test_default] is set, but neither data_reader_name or data_reader_path are.'
        assert actual == expected


def test_bad_data_filedir_15():
    try:
        tools.get_command('ray', 'exe', data_filename_test_default='e', check_executable_existence=False)
    except Exception as e:
        actual = str(e)
        expected = 'Invalid Usage: At least one of [data_filedir_train_default, data_filename_train_default, data_filedir_test_default, data_filename_test_default] is set, but neither data_reader_name or data_reader_path are.'
        assert actual == expected
