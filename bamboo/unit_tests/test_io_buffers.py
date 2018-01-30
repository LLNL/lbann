import os, sys, pytest

def test_partitioned_io_mnist(exe, dirname):
    slurm_cmd = 'srun'
    if os.getenv('SLURM_NNODES') is None:
        slurm_cmd = 'salloc -N1 ' + slurm_cmd

    max_mb=300
    for b in [300, 150, 100, 75, 60, 50]:
        for k in [1, 2, 3, 4, 5, 6]:
            num_ranks = k * max_mb / b;
            CMD='%s -n%d %s --model=%s/model_zoo/tests/model_mnist_partitioned_io.prototext --optimizer=%s/model_zoo/optimizers/opt_adagrad.prototext --reader=%s/model_zoo/data_readers/data_reader_mnist.prototext --mini_batch_size=%d --num_epochs=5 --procs_per_model=%d' % (slurm_cmd, num_ranks, exe, dirname, dirname, dirname, b, k)
            print CMD
            return_code_nockpt = os.system(CMD)
            if return_code_nockpt != 0:
                sys.stderr.write('Testing Partitioned I/O execution failed, exiting with error')
                sys.exit(1)

    diff_test = 1
    assert diff_test == 0


def test_partitioned_io_mnist(exe, dirname):
    slurm_cmd = 'srun'
    if os.getenv('SLURM_NNODES') is None:
        slurm_cmd = 'salloc -N1 ' + slurm_cmd

    max_mb=300
    for b in [300, 150, 100, 75, 60, 50]:
        for k in [1, 2, 3, 4, 5, 6]:
            num_ranks = k * max_mb / b;
            CMD='%s -n%d %s --model=%s/model_zoo/tests/model_mnist_partitioned_io.prototext --optimizer=%s/model_zoo/optimizers/opt_adagrad.prototext --reader=%s/model_zoo/data_readers/data_reader_mnist.prototext --mini_batch_size=%d --num_epochs=5 --procs_per_model=%d' % (slurm_cmd, num_ranks, exe, dirname, dirname, dirname, b, k)
            print CMD
            return_code_nockpt = os.system(CMD)
            if return_code_nockpt != 0:
                sys.stderr.write('Testing Distributed I/O execution failed, exiting with error')
                sys.exit(1)

    diff_test = 1
    assert diff_test == 0

def test_distributed_io_mnist(exe, dirname):
    slurm_cmd = 'srun'
    if os.getenv('SLURM_NNODES') is None:
        slurm_cmd = 'salloc -N1 ' + slurm_cmd

    max_mb=300
    for b in [300, 150, 100, 75, 60, 50]:
        for k in [1, 2, 3, 4, 5, 6]:
#            CMD="srun -n$((${k}*${MAX_MB}/${b})) ${FULLSCRIPT} ${STD_OPTS}"
            num_ranks = k * max_mb / b;
            CMD='%s -n%d %s --model=%s/model_zoo/tests/model_mnist_distributed_io.prototext --optimizer=%s/model_zoo/optimizers/opt_adagrad.prototext --reader=%s/model_zoo/data_readers/data_reader_mnist.prototext --mini_batch_size=%d --num_epochs=5 --procs_per_model=%d' % (slurm_cmd, num_ranks, exe, dirname, dirname, dirname, b, k)
            print CMD
            return_code_nockpt = os.system(CMD)
            if return_code_nockpt != 0:
                sys.stderr.write('Testing Partitioned I/O execution failed, exiting with error')
                sys.exit(1)

    diff_test = 1
    assert diff_test == 0
