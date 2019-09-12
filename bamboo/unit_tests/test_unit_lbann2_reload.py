import sys
sys.path.insert(0, '../common_python')
import tools
import pytest
import os


def skeleton_lbann2_reload(cluster, executables, dir_name, compiler_name,
                           weekly, data_reader_percent):
    if compiler_name not in executables:
      e = 'skeleton_lbann2_reload: default_exes[%s] does not exist' % compiler_name
      print('Skip - ' + e)
      pytest.skip(e)
    lbann2 = executables[compiler_name] + '2'

    # No checkpointing, printing weights to files.
    model_path = '{../../model_zoo/models/lenet_mnist/model_lenet_mnist.prototext,../../model_zoo/tests/model_lenet_mnist_lbann2ckpt.prototext}'
    output_file_name = '%s/bamboo/unit_tests/output/lbann2_no_checkpoint_%s_output.txt' % (dir_name, compiler_name)
    error_file_name  = '%s/bamboo/unit_tests/error/lbann2_no_checkpoint_%s_error.txt' % (dir_name, compiler_name)
    no_ckpt_dir = 'ckpt_lbann2_reload/lbann2_no_ckpt_{c}'.format(c=compiler_name)
    command = tools.get_command(
        cluster=cluster, executable=lbann2, num_nodes=1, num_processes=2,
        data_reader_name='mnist',
        data_filedir_default='/p/lscratchh/brainusr/datasets/MNIST',
        dir_name=dir_name,
        data_reader_percent=data_reader_percent,
        ckpt_dir=no_ckpt_dir,
        model_path=model_path,
        optimizer_name='sgd',
        num_epochs=2,
        output_file_name=output_file_name,
        error_file_name=error_file_name, weekly=weekly)

    return_code_no_ckpt = os.system(command)
    tools.assert_success(return_code_no_ckpt, error_file_name)

    # Run to checkpoint, printing weights to files.
    output_file_name = '%s/bamboo/unit_tests/output/lbann2_checkpoint_%s_output.txt' % (dir_name, compiler_name)
    error_file_name  = '%s/bamboo/unit_tests/error/lbann2_checkpoint_%s_error.txt' % (dir_name, compiler_name)
    ckpt_dir = 'ckpt_lbann2_reload/lbann2_ckpt_{c}'.format(c=compiler_name)
    command = tools.get_command(
        cluster=cluster, executable=lbann2, num_nodes=1, num_processes=2,
        dir_name=dir_name,
        data_filedir_default='/p/lscratchh/brainusr/datasets/MNIST',
        data_reader_name='mnist', data_reader_percent=data_reader_percent,
        ckpt_dir=ckpt_dir, model_folder='tests',
        model_name='lenet_mnist_ckpt', num_epochs=2, optimizer_name='sgd',
        output_file_name=output_file_name,
        error_file_name=error_file_name, weekly=weekly)
    return_code_ckpt_1 = os.system(command)
    tools.assert_success(return_code_ckpt_1, error_file_name)

    # Pick up from checkpoint, printing weights to files.
    output_file_name = '%s/bamboo/unit_tests/output/lbann2_restart_%s_output.txt' % (dir_name, compiler_name)
    error_file_name  = '%s/bamboo/unit_tests/error/lbann2_restart_%s_error.txt' % (dir_name, compiler_name)
    command = tools.get_command(
        cluster=cluster, executable=lbann2, num_nodes=1, num_processes=2,
        dir_name=dir_name,
        data_filedir_default='/p/lscratchh/brainusr/datasets/MNIST',
        data_reader_name='mnist',
        data_reader_percent=data_reader_percent,
        ckpt_dir=ckpt_dir,
        model_path='../../model_zoo/tests/model_lenet_mnist_lbann2ckpt.prototext',
        num_epochs=2, optimizer_name='sgd',
        output_file_name=output_file_name,
        error_file_name=error_file_name, weekly=weekly)
    return_code_ckpt_2 = os.system(command)
    tools.assert_success(return_code_ckpt_2, error_file_name)
#    os.system('rm lbann2_ckpt/model0-epoch*')
#    os.system('rm lbann2_nockpt/model0-epoch*')

    diff_result = os.system('diff -rq {ckpt} {no_ckpt}'.format(
        ckpt=ckpt_dir, no_ckpt=no_ckpt_dir))
    allow_epsilon_diff = False
    if allow_epsilon_diff and (diff_result != 0):
        equal_within_epsilon = True
        ckpt_files = os.listdir('lbann2_ckpt')
        for file_name in ckpt_files:
            ckpt_file = open('lbann2_ckpt/' + file_name, 'r')
            no_ckpt_file = open('lbann2_nockpt/' + file_name, 'r')
            for ckpt_line in ckpt_file:
                no_ckpt_line = next(no_ckpt_file)
                if ckpt_line != no_ckpt_line:
                    error_string = ('ckpt_line={ckpt_line},'
                                    ' nockpt_line={no_ckpt_line}').format(
                        ckpt_line=ckpt_line, no_ckpt_line=no_ckpt_line)
                    try:
                        ckpt_values = list(map(float, ckpt_line.split()))
                        no_ckpt_values = list(map(float, no_ckpt_line.split()))
                        num = len(ckpt_values)
                        if len(no_ckpt_values) == num:
                            for i in range(num):
                                if abs(ckpt_values[i] - no_ckpt_values[i]) > 0.5:
                                    # Not equal within epsilon.
                                    equal_within_epsilon = False
                                    print(error_string)
                        else:
                            # Length of lists don't match.
                            equal_within_epsilon = False
                            print(error_string)
                    except ValueError:
                        # Non-numerical diff.
                        equal_within_epsilon = False
                        print(error_string)
        if equal_within_epsilon:
            diff_result = 0
    path_prefix = '{d}/bamboo/unit_tests'.format(d=dir_name)
    if diff_result != 0:
        raise AssertionError(
            'diff_test={dt}\nCompare {ncd} and {cd} in {p}'.format(
                dt=diff_result, ncd=no_ckpt_dir, cd=ckpt_dir, p=path_prefix))
    assert diff_result == 0


def test_unit_lbann2_reload_clang6(cluster, exes, dirname, weekly, data_reader_percent):
    skeleton_lbann2_reload(cluster, exes, dirname, 'clang6',
                           weekly, data_reader_percent)


def test_unit_lbann2_reload_gcc7(cluster, exes, dirname, weekly, data_reader_percent):
    skeleton_lbann2_reload(cluster, exes, dirname, 'gcc7', weekly, data_reader_percent)


def test_unit_lbann2_reload_intel19(cluster, exes, dirname,
                                    weekly, data_reader_percent):
    skeleton_lbann2_reload(cluster, exes, dirname, 'intel19',
                           weekly, data_reader_percent)


# Run with python3 -m pytest -s test_unit_lbann2_reload.py -k 'test_unit_lbann2_reload_exe' --exe=<executable>
def test_unit_lbann2_reload_exe(cluster, dirname, exe, weekly, data_reader_percent):
    if exe is None:
        e = 'test_unit_lbann2_reload_exe: Non-local testing'
        print('Skip - ' + e)
        pytest.skip(e)
    exes = {'exe': exe}
    skeleton_lbann2_reload(cluster, exes, dirname, 'exe', weekly, data_reader_percent)
