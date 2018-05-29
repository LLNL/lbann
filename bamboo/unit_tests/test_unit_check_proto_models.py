import sys
sys.path.insert(0, '../common_python')
import tools
import pytest
import os, re, subprocess, sys

def skeleton_models(cluster, dir_name, executables, compiler_name):
    if compiler_name not in executables:
      pytest.skip('default_exes[%s] does not exist' % compiler_name)
    opt = 'adagrad'
    defective_models = []
    working_models = []
    tell_Dylan = []
    for subdir, dirs, files in os.walk(dir_name + '/model_zoo/models/'):
        if 'greedy' in subdir:
            print('Skipping greedy_layerwise_autoencoder_mnist, kills bamboo agent')
            continue            
        for file_name in files:
            if file_name.endswith('.prototext') and "model" in file_name:
                model_path = subdir + '/' + file_name
                print('Attempting model setup for: ' + file_name )
                data_filedir_default = None
                data_filedir_train_default=None
                data_filename_train_default=None
                data_filedir_test_default=None
                data_filename_test_default=None
                if 'mnist' in file_name:
                    data_filedir_default = '/p/lscratchf/brainusr/datasets/MNIST'
                    data_reader_name = 'mnist'
                elif 'net' in file_name:
                    data_filedir_train_default = '/p/lscratche/brainusr/datasets/ILSVRC2012/original/train/'
                    data_filename_train_default = '/p/lscratche/brainusr/datasets/ILSVRC2012/labels/train.txt'
                    data_filedir_test_default = '/p/lscratche/brainusr/datasets/ILSVRC2012/original/val/'
                    data_filename_test_default = '/p/lscratche/brainusr/datasets/ILSVRC2012/original/labels/val.txt'
                    data_reader_name = 'imagenet'
                elif 'cifar' in file_name:
                    data_filename_train_default = '/p/lscratchf/brainusr/datasets/cifar10-bin/data_all.bin'
                    data_filename_test_default = '/p/lscratchf/brainusr/datasets/cifar10-bin/test_batch.bin'
                    data_reader_name = 'cifar10'
                elif 'char' in file_name:
                    data_filedir_default = '/p/lscratche/brainusr/datasets/tinyshakespeare/'
                    data_reader_name = 'ascii'
                else:
                    print("Tell Dylan which data reader this model needs")
                    tell_Dylan.append(file_name)
                    continue
                if (cluster == 'ray') and (data_reader_name in ['cifar10', 'ascii']):
                    print('Skipping %s because data is not available on ray' % model_path)
                else:
                    output_file_name = '%s/bamboo/unit_tests/output/check_proto_models_%s_%s_output.txt' % (dir_name, file_name, compiler_name)
                    error_file_name = '%s/bamboo/unit_tests/error/check_proto_models_%s_%s_error.txt' % (dir_name, file_name, compiler_name)
                    cmd = tools.get_command(
                        cluster=cluster, executable=executables[compiler_name], num_nodes=1,
                        partition='pdebug', time_limit=1, dir_name=dir_name,
                        data_filedir_default=data_filedir_default,
                        data_filedir_train_default=data_filedir_train_default,
                        data_filename_train_default=data_filename_train_default,
                        data_filedir_test_default=data_filedir_test_default,
                        data_filename_test_default=data_filename_test_default,
                        data_reader_name=data_reader_name, exit_after_setup=True,
                        model_path=model_path, optimizer_name=opt,
                        output_file_name=output_file_name, error_file_name=error_file_name)
                    if os.system(cmd) != 0:
                        print("Error detected in " + model_path)
                        #defective_models.append(file_name)
                        defective_models.append(cmd)
                    else:
                       working_models.append(cmd)
    num_defective = len(defective_models)
    if num_defective != 0:
        print('Working models: %d. Defective models: %d', len(working_models), num_defective)
        print('Errors for: The following models exited with errors')
        for model in defective_models:
            print(model)
        print('Errors for: tell Dylan: the following models have unknown data readers:')
        for model in tell_Dylan :
            print(model)
    assert num_defective == 0

def test_unit_models_clang4(cluster, dirname, exes):
    skeleton_models(cluster, dirname, exes, 'clang4')

def test_unit_models_gcc4(cluster, dirname, exes):
    if cluster in ['catalyst', 'quartz', 'surface']:
        pytest.skip('FIXME')
        # Catalyst Errors:
        # assert 4 == 0
        # Surface Errors:
        # assert 8 == 0
    skeleton_models(cluster, dirname, exes, 'gcc4')

def test_unit_models_gcc7(cluster, dirname, exes):
    skeleton_models(cluster, exes, dirname, 'gcc7')

def test_unit_models_intel18(cluster, dirname, exes):
    skeleton_models(cluster, dirname, exes, 'intel18')

# Run with python -m pytest -s test_unit_check_proto_models.py -k 'test_unit_models_exe' --exe=<executable>
def test_unit_lbann2_reload_exe(cluster, dirname, exe):
    if exe == None:
        pytest.skip('Non-local testing')
    exes = {'exe' : exe}
    skeleton_models(cluster, exes, dirname, 'exe')
