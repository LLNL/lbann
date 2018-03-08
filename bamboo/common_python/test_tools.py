import pytest
import tools

# This test isn't in a directory to be run from Bamboo
# Run locally with python -m pytest -s

def test_command_catalyst():
    actual = tools.get_command(cluster='catalyst', executable='exe', num_nodes=20, partition='pdebug', time_limit=30, num_processes=40, dir_name='dir', data_filedir_ray='filedir', data_reader_name='mnist', data_reader_percent=0.10, exit_after_setup=True, mini_batch_size=15, model_folder='models/folder', model_name='lenet', num_epochs=7, optimizer_name='adagrad', processes_per_model=10, output_file_name='output_file')
    expected = 'salloc --nodes=20 --partition=pdebug --time=30 srun --ntasks=40 exe --reader=dir/model_zoo/data_readers/data_reader_mnist.prototext --data_reader_percent=0.100000 --exit_after_setup --mini_batch_size=15 --model=dir/model_zoo/models/folder/model_lenet.prototext --num_epochs=7 --optimizer=dir/model_zoo/optimizers/opt_adagrad.prototext --procs_per_model=10 > output_file'
    assert actual == expected

def test_command_surface():
    actual = tools.get_command(cluster='surface', executable='exe', num_nodes=20, partition='pdebug', time_limit=30, num_processes=40, dir_name='dir', data_filedir_ray='filedir', data_reader_name='mnist', data_reader_percent=0.10, exit_after_setup=True, mini_batch_size=15, model_folder='models/folder', model_name='lenet', num_epochs=7, optimizer_name='adagrad', processes_per_model=10, output_file_name='output_file')
    expected = 'salloc --nodes=20 --partition=pbatch --time=30 srun --ntasks=40 exe --reader=dir/model_zoo/data_readers/data_reader_mnist.prototext --data_reader_percent=0.100000 --exit_after_setup --mini_batch_size=15 --model=dir/model_zoo/models/folder/model_lenet.prototext --num_epochs=7 --optimizer=dir/model_zoo/optimizers/opt_adagrad.prototext --procs_per_model=10 > output_file'
    assert actual == expected

def test_command_ray():
    actual = tools.get_command(cluster='ray', executable='exe', num_nodes=20, partition='pdebug', time_limit=30, num_processes=40, dir_name='dir', data_filedir_ray='filedir', data_reader_name='mnist', data_reader_percent=0.10, exit_after_setup=True, mini_batch_size=15, model_folder='models/folder', model_name='lenet', num_epochs=7, optimizer_name='adagrad', processes_per_model=10, output_file_name='output_file')
    expected = 'bsub -x -G guests -Is -n 40 -q pdebug -R "span[ptile=2]" -W 30 mpirun -np 40 -N 2 exe --data_filedir=filedir --reader=dir/model_zoo/data_readers/data_reader_mnist.prototext --data_reader_percent=0.100000 --exit_after_setup --mini_batch_size=15 --model=dir/model_zoo/models/folder/model_lenet.prototext --num_epochs=7 --optimizer=dir/model_zoo/optimizers/opt_adagrad.prototext --procs_per_model=10 > output_file'
    assert actual == expected
