import os, pytest

def test_gradient_check(exe, dirname):
    slurm_command = 'srun'
    if os.getenv('SLURM_NNODES') is None:
        slurm_command = 'salloc -N1 ' + slurm_command
    return_code = os.system('%s --ntasks=1 %s --model=%s/model_zoo/tests/model_mnist_ridge_regression.prototext --optimizer=%s/model_zoo/optimizers/opt_adam.prototext --reader=%s/model_zoo/data_readers/data_reader_mnist.prototext' % (slurm_command, exe, dirname, dirname, dirname))
    assert return_code == 0
