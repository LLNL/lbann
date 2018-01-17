import os, pytest

def test_checkpoint_lenet(exe, dirname):
    slurm_cmd = 'srun -n1'
    if os.getenv('SLURM_NNODES') is None:
        slurm_cmd = 'salloc -N1 ' + slurm_cmd

    return_code_nockpt = os.system('%s %s --model=%s/model_zoo/tests/model_lenet_mnist_ckpt.prototext --optimizer=%s/model_zoo/optimizers/opt_sgd.prototext --reader=%s/model_zoo/data_readers/data_reader_mnist.prototext --num_epochs=2' % (slurm_cmd, exe, dirname, dirname, dirname))
    if return_code_nockpt != 0:
        sys.stderr.write('LeNet (no checkpoint) execution failed, exiting with error')
        sys.exit(1)
    os.system('mv ckpt ckpt_baseline')

    return_code_ckpt_1 = os.system('%s %s --model=%s/model_zoo/tests/model_lenet_mnist_ckpt.prototext --optimizer=%s/model_zoo/optimizers/opt_sgd.prototext --reader=%s/model_zoo/data_readers/data_reader_mnist.prototext --num_epochs=1' % (slurm_cmd, exe, dirname, dirname, dirname))

    if return_code_ckpt_1 != 0:
        sys.stderr.write('LeNet (checkpoint) execution failed, exiting with error')
        sys.exit(1)

    return_code_ckpt_2 = os.system('%s %s --model=%s/model_zoo/tests/model_lenet_mnist_ckpt.prototext --optimizer=%s/model_zoo/optimizers/opt_sgd.prototext --reader=%s/model_zoo/data_readers/data_reader_mnist.prototext --num_epochs=2' % (slurm_cmd, exe, dirname, dirname, dirname))

    if return_code_ckpt_2 != 0:
        sys.stderr.write('LeNet execution (restart from checkpoint) failed, exiting with error')
        sys.exit(1)

    diff_test = os.system('diff ckpt/shared.epoch.2.step.1688 ckpt_baseline/shared.epoch.2.step.1688')
    os.system('rm -rf ckpt*')
    assert diff_test == 0
