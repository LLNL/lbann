import pytest, subprocess

def test_gradient_check(exe, dirname):
    return_code = subprocess.call('salloc -N1 srun --ntasks=1 %s --model=%s/model_zoo/tests/model_mnist_ridge_regression.prototext --optimizer=%s/model_zoo/optimizers/opt_adam.prototext --reader=%s/model_zoo/data_readers/data_reader_mnist.prototext' % (exe, dirname, dirname, dirname))
    assert return_code == 0
