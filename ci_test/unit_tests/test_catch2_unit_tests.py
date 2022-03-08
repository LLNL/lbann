import sys
sys.path.insert(0, '../common_python')
import tools
import pytest
import os, re
import subprocess as sp

def hack_find_spack_build_dir(basedir):
    with os.scandir(basedir) as it:
        for entry in it:
            if entry.is_dir() and re.match(r'spack-.*', entry.name):
                return entry.path

def get_system_seq_launch(cluster):
    if cluster in ['lassen', 'ray']:
        return ['lrun', '-1', '--smpiargs=\"-disable_gpu_hooks\"']
    return ['srun', '-N1', '-n1', '--mpibind=off']

def get_system_mpi_launch(cluster):
    if cluster in ['lassen', 'ray']:
        return ['jsrun', '-n2', '-r1', '-a4', '-c', 'ALL_CPUS', '-g', 'ALL_GPUS', '-d', 'packed', '-b', 'packed:10']
    elif cluster == 'pascal':
        return ['srun', '-N2', '--ntasks-per-node=2', '--mpibind=off']
    else: # Corona and Catalyst
        return ['srun', '-N2', '--ntasks-per-node=4']

# Notice that these tests will automatically skip if the executable
# doesn't exist. Since we do not save the testing executable as a
# GitLab CI artifact on Catalyst, Corona, or Pascal, this should only
# run on Ray and Lassen in GitLab CI testing pipelines.
def test_run_sequential_catch_tests(cluster, dirname):
    output_dir = os.path.join(dirname, 'ci_test', 'unit_tests')
    build_dir = hack_find_spack_build_dir(dirname)
    seq_catch_exe = os.path.join(build_dir, 'unit_test', 'seq-catch-tests')
    if not os.path.exists(seq_catch_exe):
        print('Skip - executable not found')
        pytest.skip('executable not found')
    # Run the sequential tests
    seq_launch = get_system_seq_launch(cluster)
    seq_output_file_name = 'seq_catch_tests_output-%s.xml' % (cluster)
    seq_output_file = os.path.join(output_dir, seq_output_file_name)
    seq_error_file = os.path.join(output_dir, "error", "seq-catch-test-error.log")
    seq_catch_args = [seq_catch_exe, '-r', 'junit', '-o', seq_output_file]
    output = sp.run(seq_launch + seq_catch_args)
    tools.assert_success(output.returncode, seq_error_file)

def test_run_parallel_catch_tests(cluster, dirname):
    output_dir = os.path.join(dirname, 'ci_test', 'unit_tests')
    build_dir = hack_find_spack_build_dir(dirname)
    mpi_catch_exe = os.path.join(build_dir, 'unit_test', 'mpi-catch-tests')
    if not os.path.exists(mpi_catch_exe):
        print('Skip - executable not found')
        pytest.skip('executable not found')
    # Run the parallel tests
    mpi_launch = get_system_mpi_launch(cluster)
    mpi_output_file_name = 'mpi_catch_tests_output-%s-rank=%%r-size=%%s.xml' % (cluster)
    mpi_output_file = os.path.join(output_dir, mpi_output_file_name)
    mpi_error_file = os.path.join(output_dir, "error", "mpi-catch-test-error.log")
    mpi_catch_args = [mpi_catch_exe, '-r', 'junit', '-o', mpi_output_file]
    output = sp.run(mpi_launch + mpi_catch_args)
    tools.assert_success(output.returncode, mpi_error_file)

def test_run_parallel_filesystem_catch_tests(cluster, dirname):
    output_dir = os.path.join(dirname, 'ci_test', 'unit_tests')
    build_dir = hack_find_spack_build_dir(dirname)
    mpi_catch_exe = os.path.join(build_dir, 'unit_test', 'mpi-catch-tests')
    if not os.path.exists(mpi_catch_exe):
        print('Skip - executable not found')
        pytest.skip('executable not found')
    # Run the parallel tests
    mpi_launch = get_system_mpi_launch(cluster)
    mpi_output_file_name = 'mpi_filesystem_catch_tests_output-%s-rank=%%r-size=%%s.xml' % (cluster)
    mpi_output_file = os.path.join(output_dir, mpi_output_file_name)
    mpi_error_file = os.path.join(output_dir, "error", "mpi-filesystem-catch-test-error.log")
    mpi_catch_args = [mpi_catch_exe, '"[filesystem]"', '-r', 'junit', '-o', mpi_output_file]
    output = sp.run(mpi_launch + mpi_catch_args)
    tools.assert_success(output.returncode, mpi_error_file)
