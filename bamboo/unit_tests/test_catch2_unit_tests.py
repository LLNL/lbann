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
        return ['jsrun', '-n2', '-r1', '-a4', '-c40', '-g4', '-d', 'packed', '-b', 'packed:10']
    elif cluster == 'pascal':
        return ['srun', '-N2', '--ntasks-per-node=2', '--mpibind=off']
    else: # Corona and Catalyst
        return ['srun', '-N2', '--ntasks-per-node=4']

def test_run_sequential_catch_tests(cluster, dirname):
    output_dir = os.path.join(dirname, 'bamboo', 'unit_tests')
    build_dir = hack_find_spack_build_dir(dirname)
    seq_catch_exe = os.path.join(build_dir, 'unit_test', 'seq-catch-tests')
    if not os.path.exists(seq_catch_exe):
        print('Skip - executable not found')
        pytest.skip('executable not found')
    # Run the sequential tests
    seq_launch = get_system_seq_launch(cluster)
    seq_output_file_name = 'seq_catch_tests_output-%s.xml' % (cluster)
    seq_output_file = os.path.join(output_dir, seq_output_file_name)
    seq_catch_args = [seq_catch_exe, '-r', 'junit', '-o', seq_output_file]
    output = sp.run(seq_launch + seq_catch_args)
    tools.assert_success(output.returncode, seq_output_file)

def test_run_parallel_catch_tests(cluster, dirname):
    output_dir = os.path.join(dirname, 'bamboo', 'unit_tests')
    build_dir = hack_find_spack_build_dir(dirname)
    mpi_catch_exe = os.path.join(build_dir, 'unit_test', 'mpi-catch-tests')
    if not os.path.exists(mpi_catch_exe):
        print('Skip - executable not found')
        pytest.skip('executable not found')
    # Run the parallel tests
    mpi_launch = get_system_mpi_launch(cluster)
    mpi_output_file_name = 'mpi_catch_tests_output-%s-rank=%%r-size=%%s.xml' % (cluster)
    mpi_output_file = os.path.join(output_dir, mpi_output_file_name)
    mpi_catch_args = [mpi_catch_exe, '-r', 'junit', '-o', mpi_output_file]
    output = sp.run(mpi_launch + mpi_catch_args)
    tools.assert_success(output.returncode, mpi_output_file)
    
