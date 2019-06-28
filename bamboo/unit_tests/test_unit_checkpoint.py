import sys
sys.path.insert(0, '../common_python')
import tools
import pytest
import os


def skeleton_checkpoint_lenet_shared(cluster, executables, dir_name,
                                     compiler_name):
    if compiler_name not in executables:
        e = 'skeleton_checkpoint_lenet_shared: default_exes[%s] does not exist' % compiler_name
        print('Skip - ' + e)
        pytest.skip(e)
    exe = executables[compiler_name]

    # No checkpointing, printing weights to files.
    output_file_name = '%s/bamboo/unit_tests/output/checkpoint_lenet_shared_no_checkpoint_%s_output.txt' % (dir_name, compiler_name)
    error_file_name  = '%s/bamboo/unit_tests/error/checkpoint_lenet_shared_no_checkpoint_%s_error.txt' % (dir_name, compiler_name)
    command = tools.get_command(
        cluster=cluster, executable=exe, num_nodes=1, num_processes=2,
        dir_name=dir_name,
        data_filedir_default='/p/lscratchh/brainusr/datasets/MNIST',
        data_reader_name='mnist', model_folder='tests',
        model_name='lenet_mnist_ckpt', num_epochs=2, optimizer_name='sgd',
        output_file_name=output_file_name, error_file_name=error_file_name)
    return_code_nockpt = os.system(command)
    if return_code_nockpt != 0:
        sys.stderr.write('LeNet (no checkpoint) execution failed, exiting with error')
        sys.exit(1)
    ckpt_pre = 'ckpt_pre_lenet_shared_{c}'.format(c=compiler_name)
    os.system('mv ckpt {c}'.format(c=ckpt_pre))

    # Run to checkpoint, printing weights to files.
    output_file_name = '%s/bamboo/unit_tests/output/checkpoint_lenet_shared_checkpoint_%s_output.txt' % (dir_name, compiler_name)
    error_file_name  = '%s/bamboo/unit_tests/error/checkpoint_lenet_shared_checkpoint_%s_error.txt' % (dir_name, compiler_name)
    command = tools.get_command(
        cluster=cluster, executable=exe, num_nodes=1, num_processes=2,
        dir_name=dir_name,
        data_filedir_default='/p/lscratchh/brainusr/datasets/MNIST',
        data_reader_name='mnist', model_folder='tests',
        model_name='lenet_mnist_ckpt', num_epochs=1, optimizer_name='sgd',
        output_file_name=output_file_name, error_file_name=error_file_name)
    return_code_ckpt_1 = os.system(command)
    if return_code_ckpt_1 != 0:
        sys.stderr.write('LeNet (checkpoint) execution failed, exiting with error')
        sys.exit(1)

    # Pick up from checkpoint, printing weights to files.
    output_file_name = '%s/bamboo/unit_tests/output/checkpoint_lenet_shared_restart_%s_output.txt' % (dir_name, compiler_name)
    error_file_name  = '%s/bamboo/unit_tests/error/checkpoint_lenet_shared_restart_%s_error.txt' % (dir_name, compiler_name)
    command = tools.get_command(
        cluster=cluster, executable=exe, num_nodes=1, num_processes=2,
        dir_name=dir_name,
        data_filedir_default='/p/lscratchh/brainusr/datasets/MNIST',
        data_reader_name='mnist', model_folder='tests',
        model_name='lenet_mnist_ckpt', num_epochs=2, optimizer_name='sgd',
        output_file_name=output_file_name, error_file_name=error_file_name)
    return_code_ckpt_2 = os.system(command)
    if return_code_ckpt_2 != 0:
        sys.stderr.write('LeNet execution (restart from checkpoint) failed, exiting with error')
        sys.exit(1)

    diff_test = os.system('diff -rq ckpt {c}'.format(c=ckpt_pre))
    os.system('mv ckpt ckpt_post_lenet_shared_{c}'.format(c=compiler_name))
    assert diff_test == 0


def skeleton_checkpoint_lenet_distributed(cluster, executables, dir_name,
                                          compiler_name):
     if compiler_name not in executables:
         e = 'skeleton_checkpoint_lenet_distributed: default_exes[%s] does not exist' % compiler_name
         print('Skip - ' + e)
         pytest.skip(e)
     exe = executables[compiler_name]

     # No checkpointing, printing weights to files.
     output_file_name = '%s/bamboo/unit_tests/output/checkpoint_lenet_distributed_no_checkpoint_%s_output.txt' % (dir_name, compiler_name)
     error_file_name  = '%s/bamboo/unit_tests/error/checkpoint_lenet_distributed_no_checkpoint_%s_error.txt' % (dir_name, compiler_name)
     command = tools.get_command(
         cluster=cluster, executable=exe, num_nodes=1, num_processes=2,
         dir_name=dir_name,
         data_filedir_default='/p/lscratchh/brainusr/datasets/MNIST',
         data_reader_name='mnist', model_folder='tests',
         model_name='lenet_mnist_dist_ckpt', num_epochs=2, optimizer_name='sgd',
        output_file_name=output_file_name, error_file_name=error_file_name)
     return_code_nockpt = os.system(command)
     if return_code_nockpt != 0:
         sys.stderr.write('LeNet (no checkpoint) execution failed, exiting with error')
         sys.exit(1)
     ckpt_pre = 'ckpt_pre_lenet_distributed_{c}'.format(c=compiler_name)
     os.system('mv ckpt {c}'.format(c=ckpt_pre))

     # Run to checkpoint, printing weights to files.
     output_file_name = '%s/bamboo/unit_tests/output/checkpoint_lenet_distributed_checkpoint_%s_output.txt' % (dir_name, compiler_name)
     error_file_name  = '%s/bamboo/unit_tests/error/checkpoint_lenet_distributed_checkpoint_%s_error.txt' % (dir_name, compiler_name)
     command = tools.get_command(
         cluster=cluster, executable=exe, num_nodes=1, num_processes=2,
         dir_name=dir_name,
         data_filedir_default='/p/lscratchh/brainusr/datasets/MNIST',
         data_reader_name='mnist', model_folder='tests',
         model_name='lenet_mnist_dist_ckpt', num_epochs=1, optimizer_name='sgd',
        output_file_name=output_file_name, error_file_name=error_file_name)
     return_code_ckpt_1 = os.system(command)
     if return_code_ckpt_1 != 0:
         sys.stderr.write('LeNet (checkpoint) execution failed, exiting with error')
         sys.exit(1)

     # Pick up from checkpoint, printing weights to files.
     output_file_name = '%s/bamboo/unit_tests/output/checkpoint_lenet_distributed_restart_%s_output.txt' % (dir_name, compiler_name)
     error_file_name  = '%s/bamboo/unit_tests/error/checkpoint_lenet_distributed_restart_%s_error.txt' % (dir_name, compiler_name)
     command = tools.get_command(
         cluster=cluster, executable=exe, num_nodes=1, num_processes=2,
         dir_name=dir_name,
         data_filedir_default='/p/lscratchh/brainusr/datasets/MNIST',
         data_reader_name='mnist', model_folder='tests',
         model_name='lenet_mnist_dist_ckpt', num_epochs=2, optimizer_name='sgd',
        output_file_name=output_file_name, error_file_name=error_file_name)
     return_code_ckpt_2 = os.system(command)
     if return_code_ckpt_2 != 0:
         sys.stderr.write('LeNet execution (restart from checkpoint) failed, exiting with error')
         sys.exit(1)

     diff_test = os.system('diff -rq ckpt {c}'.format(c=ckpt_pre))
     os.system('mv ckpt ckpt_post_lenet_distributed_{c}'.format(c=compiler_name))
     assert diff_test == 0


def test_unit_checkpoint_lenet_clang6(cluster, exes, dirname):
    skeleton_checkpoint_lenet_shared(cluster, exes, dirname, 'clang6')
    skeleton_checkpoint_lenet_distributed(cluster, exes, dirname, 'clang6')


def test_unit_checkpoint_lenet_gcc7(cluster, exes, dirname):
    skeleton_checkpoint_lenet_shared(cluster, exes, dirname, 'gcc7')
    skeleton_checkpoint_lenet_distributed(cluster, exes, dirname, 'gcc7')


def test_unit_checkpoint_lenet_intel19(cluster, exes, dirname):
    skeleton_checkpoint_lenet_shared(cluster, exes, dirname, 'intel19')
    skeleton_checkpoint_lenet_distributed(cluster, exes, dirname, 'intel19')


# Run with python -m pytest -s test_unit_checkpoint.py -k 'test_unit_checkpoint_lenet_exe' --exe=<executable>
def test_unit_checkpoint_lenet_exe(cluster, dirname, exe):
    if exe is None:
        e = 'test_unit_checkpoint_lenet_exe: Non-local testing'
        print('Skip - ' + e)
        pytest.skip(e)
    exes = {'exe': exe}
    skeleton_checkpoint_lenet_shared(cluster, exes, dirname, 'exe')
    skeleton_checkpoint_lenet_distributed(cluster, exes, dirname, 'exe')
