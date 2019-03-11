import sys
sys.path.insert(0, '../common_python')
import tools
import os, sys

def test_unit_no_params_bad(cluster, exes):
    exe = exes['gcc4']
    sys.stderr.write('TESTING: run lbann with no params; lbann should throw exception\n')
    command = tools.get_command(
        cluster=cluster, executable=exe, exit_after_setup=True)
    return_code = os.system(command)
    assert return_code != 0


def test_unit_one_model_bad(cluster, exes):
    exe = exes['gcc4']
    sys.stderr.write('TESTING: run lbann with no optimizer or reader; lbann should throw exception\n')
    model_path = 'prototext/model_mnist_simple_1.prototext'
    command = tools.get_command(
        cluster=cluster, executable=exe, exit_after_setup=True,
        model_path=model_path)
    return_code = os.system(command)
    assert return_code != 0


def test_unit_two_models_bad(cluster, exes):
    exe = exes['gcc4']
    sys.stderr.write('TESTING: run lbann with two models but no optimizer or reader; lbann should throw exception\n')
    model_path = '{prototext/model_mnist_simple_1.prototext,prototext/model_mnist_simple_1.prototext}'
    command = tools.get_command(
        cluster=cluster, executable=exe, exit_after_setup=True,
        model_path=model_path)
    return_code = os.system(command)
    assert return_code != 0


def test_unit_two_models_bad2(cluster, exes):
    exe = exes['gcc4']
    sys.stderr.write('TESTING: run lbann with two models with missing {; lbann should throw exception\n')
    model_path='prototext/model_mnist_simple_1.prototext,prototext/model_mnist_simple_1.prototext}'
    command = tools.get_command(
        cluster=cluster, executable=exe, exit_after_setup=True,
        model_path=model_path)
    return_code = os.system(command)
    assert return_code != 0


def test_unit_missing_optimizer(cluster, exes):
    exe = exes['gcc4']
    sys.stderr.write('TESTING: run lbann with two models, reader, but no optimizer; lbann should throw exception\n')
    model_path='{prototext/model_mnist_simple_1.prototext,prototext/model_mnist_simple_1.prototext}'
    data_reader_path='prototext/data_reader_mnist.prototext'
    command = tools.get_command(
        cluster=cluster, executable=exe, data_reader_path=data_reader_path,
        data_filedir_default='/p/lscratchh/brainusr/datasets/MNIST',
        exit_after_setup=True, model_path=model_path)
    return_code = os.system(command)
    assert return_code != 0


def test_unit_missing_reader(cluster, exes):
    exe = exes['gcc4']
    sys.stderr.write('TESTING: run lbann with two models, reader, but no reader; lbann should throw exception\n')
    model_path = '{prototext/model_mnist_simple_1.prototext,prototext/model_mnist_simple_1.prototext}'
    optimizer_path = 'prototext/opt_sgd.prototext'
    command = tools.get_command(
        cluster=cluster, executable=exe, exit_after_setup=True,
        model_path=model_path, optimizer_path=optimizer_path)
    return_code = os.system(command)
    assert return_code != 0


def test_unit_bad_params(cluster, exes):
    exe = exes['gcc4']
    sys.stderr.write('TESTING: run lbann with ill-formed param (missing -) lbann should throw exception\n')
    (command_allocate, command_run, _, _) = tools.get_command(cluster=cluster, executable=exe, return_tuple=True)
    return_code = os.system('%s%s %s -exit_after_setup --reader=prototext/data_reader_mnist.prototext --model={prototext/model_mnist_simple_1.prototext,prototext/model_mnist_simple_1.prototext} --optimizer=prototext/opt_sgd.prototext' % (command_allocate, command_run, exe))
    assert return_code != 0


def test_unit_should_work(cluster, exes):
    exe = exes['gcc4']
    sys.stderr.write('TESTING: run lbann with two models, reader, and optimizer; lbann should NOT throw exception\n')
    model_path = '{prototext/model_mnist_simple_1.prototext,prototext/model_mnist_simple_1.prototext}'
    data_reader_path = 'prototext/data_reader_mnist.prototext'
    optimizer_path = 'prototext/opt_sgd.prototext'
    command = tools.get_command(
        cluster=cluster, executable=exe, data_reader_path=data_reader_path,
        data_filedir_default='/p/lscratchh/brainusr/datasets/MNIST',
        exit_after_setup=True, model_path=model_path,
        optimizer_path=optimizer_path)
    return_code = os.system(command)
    assert return_code != 0
