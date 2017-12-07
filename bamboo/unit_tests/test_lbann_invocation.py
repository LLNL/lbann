import os, pytest, sys

def test_no_params_bad(exe,dirname) :
    sys.stderr.write('TESTING: run lbann with no params; lbann should throw exception\n');
    return_code = os.system('%s --exit_after_setup' % (exe))
    assert return_code != 0

def test_one_model_bad(exe,dirname) :
    sys.stderr.write('TESTING: run lbann with no optimizer or reader; lbann should throw exception\n');
    return_code = os.system('%s --exit_after_setup --model=prototext/model_mnist_simple_1.prototext' % (exe))
    assert return_code != 0

def test_two_models_bad(exe,dirname) :
    sys.stderr.write('TESTING: run lbann with two models but no optimizer or reader; lbann should throw exception\n');
    return_code = os.system('%s --exit_after_setup --model={prototext/model_mnist_simple_1.prototext,prototext/model_mnist_simple_1.prototext}' % (exe))
    assert return_code != 0

def test_two_models_bad2(exe,dirname) :
    sys.stderr.write('TESTING: run lbann with two models with missing {; lbann should throw exception\n');
    return_code = os.system('%s --exit_after_setup --model=prototext/model_mnist_simple_1.prototext,prototext/model_mnist_simple_1.prototext}' % (exe))
    assert return_code != 0

def test_missing_optimizer(exe,dirname) :
    sys.stderr.write('TESTING: run lbann with two models, reader, but no optimizer; lbann should throw exception\n');
    return_code = os.system('%s --exit_after_setup --model={prototext/model_mnist_simple_1.prototext,prototext/model_mnist_simple_1.prototext} --reader=prototext/data_reader_mnist.prototext' % (exe))
    assert return_code != 0

def test_missing_reader(exe,dirname) :
    sys.stderr.write('TESTING: run lbann with two models, reader, but no optimizer; lbann should throw exception\n');
    return_code = os.system('%s --exit_after_setup --model={prototext/model_mnist_simple_1.prototext,prototext/model_mnist_simple_1.prototext} --optimizer=prototext/opt_sgd.prototext' % (exe))
    assert return_code != 0

def test_bad_params(exe,dirname) :
    sys.stderr.write('TESTING: run lbann with ill-formed param (missing -) lbann should throw exception\n');
    return_code = os.system('%s -exit_after_setup --reader=prototext/data_reader_mnist.prototext --model={prototext/model_mnist_simple_1.prototext,prototext/model_mnist_simple_1.prototext} --optimizer=prototext/opt_sgd.prototext' % (exe))
    assert return_code != 0

def test_should_work(exe,dirname) :
    sys.stderr.write('TESTING: run lbann with two models, reader, and optimizer; lbann should NOT throw exception\n');
    return_code = os.system('%s --exit_after_setup --reader=prototext/data_reader_mnist.prototext --model={prototext/model_mnist_simple_1.prototext,prototext/model_mnist_simple_1.prototext} --optimizer=prototext/opt_sgd.prototext' % (exe))
    assert return_code == 0
