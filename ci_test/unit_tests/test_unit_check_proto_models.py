import sys
sys.path.insert(0, '../common_python')
import tools
import pytest
import os


def skeleton_models(cluster, dir_name,
                    weekly, data_reader_fraction):
    opt = 'sgd'
    node_count = 1
    time_limit = 1
    defective_models = []
    working_models = []
    for subdir, dirs, files in os.walk(dir_name + '/model_zoo/models/'):
        for file_name in files:
            if file_name.endswith('.prototext') and "model" in file_name:
                model_path = subdir + '/' + file_name
                print('Attempting model setup for: ' + file_name)
                data_filedir_default = None
                data_filedir_train_default=None
                data_filename_train_default=None
                data_filedir_test_default=None
                data_filename_test_default=None
                data_reader_path=None
                if 'mnist' in file_name:
                    data_filedir_default = '/p/lscratchh/brainusr/datasets/MNIST'
                    data_reader_name = 'mnist'
                elif 'adversarial' in file_name:
                    data_filedir_default = '/p/lscratchh/brainusr/datasets/MNIST'
                    data_reader_path = '%s/model_zoo/models/gan/mnist/adversarial_data.prototext' % (dir_name)
                    data_reader_name = None
                elif 'discriminator' in file_name:
                    data_filedir_default = '/p/lscratchh/brainusr/datasets/MNIST'
                    data_reader_path = '%s/model_zoo/models/gan/mnist/discriminator_data.prototext' % (dir_name)
                    data_reader_name = None
                elif 'net' in file_name:
                    data_filedir_train_default = '/p/lscratchh/brainusr/datasets/ILSVRC2012/original/train/'
                    data_filename_train_default = '/p/lscratchh/brainusr/datasets/ILSVRC2012/labels/train.txt'
                    data_filedir_test_default = '/p/lscratchh/brainusr/datasets/ILSVRC2012/original/val/'
                    data_filename_test_default = '/p/lscratchh/brainusr/datasets/ILSVRC2012/labels/val.txt'
                    data_reader_name = 'imagenet'
                    node_count = 2
                    if cluster == 'ray':
                        time_limit = 3
                    if 'resnet50' in file_name:
                        node_count = 8
                        if not weekly:
                            continue # This is too many nodes for nightly.
                elif 'cifar' in file_name:
                    data_filename_train_default = '/p/lscratchh/brainusr/datasets/cifar10-bin/data_all.bin'
                    data_filename_test_default = '/p/lscratchh/brainusr/datasets/cifar10-bin/test_batch.bin'
                    data_reader_name = 'cifar10'
                elif 'char' in file_name:
                    data_filedir_default = '/p/lscratchh/brainusr/datasets/tinyshakespeare/'
                    data_reader_name = 'ascii'
                else:
                    print(
                        "No access to dataset that model={m} requires.".format(
                            m=file_name))
                    continue
                if (cluster == 'ray') and \
                        (data_reader_name in ['cifar10', 'ascii']):
                    print('Skipping %s because data is not available on ray' % model_path)
                elif (cluster == 'ray') or (cluster == 'pascal') and \
                        ('conv_autoencoder' in file_name) or ('gan' in subdir):
                    print('Skipping %s because unpooling/noise is not implemented on gpu' % model_path)
                else:
                    output_file_name = '%s/ci_test/unit_tests/output/check_proto_models_%s_output.txt' % (dir_name, file_name)
                    error_file_name = '%s/ci_test/unit_tests/error/check_proto_models_%s_error.txt' % (dir_name, file_name)
                    cmd = tools.get_command(
                        cluster=cluster,
                        num_nodes=node_count,
                        partition='pbatch', time_limit=time_limit,
                        dir_name=dir_name,
                        data_filedir_default=data_filedir_default,
                        data_filedir_train_default=data_filedir_train_default,
                        data_filename_train_default=data_filename_train_default,
                        data_filedir_test_default=data_filedir_test_default,
                        data_filename_test_default=data_filename_test_default,
                        data_reader_name=data_reader_name,
                        data_reader_path=data_reader_path,
                        data_reader_fraction=data_reader_fraction,
                        exit_after_setup=True, model_path=model_path,
                        optimizer_name=opt,
                        output_file_name=output_file_name,
                        error_file_name=error_file_name, weekly=weekly)
                    if os.system(cmd) != 0:
                        print("Error detected in " + model_path)
                        #defective_models.append(file_name)
                        defective_models.append(cmd)
                    else:
                       working_models.append(cmd)
    num_defective = len(defective_models)
    if num_defective != 0:
        print('Working models: %d. Defective models: %d' % (
            len(working_models), num_defective))
        print('Errors for: The following models exited with errors')
        for model in defective_models:
            print(model)
    if num_defective != 0:
        raise AssertionError(
            'num_defective={nd}\nDefective models:\n{dms}'.format(
                nd=num_defective, dms=defective_models))


# Run with python3 -m pytest -s test_unit_check_proto_models.py -k 'test_unit_models'
def test_unit_models(cluster, dirname, weekly, data_reader_fraction):
    skeleton_models(cluster, dirname, weekly, data_reader_fraction)
