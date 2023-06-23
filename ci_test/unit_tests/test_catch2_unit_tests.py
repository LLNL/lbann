import sys
sys.path.insert(0, '../common_python')
import tools
import pytest
import os, re
import subprocess as sp

# BVE this should work for now, but needs to be cleaned up more
def hack_find_spack_build_dir(basedir):
    if os.getenv('LBANN_BUILD_DIR', default=None):
        build_dir = os.getenv('LBANN_BUILD_DIR')
        return basedir + '/' + build_dir
    else:
        build_dir = basedir + '/builds'
        with os.scandir(basedir) as it:
            for entry in it:
                if entry.is_dir() and re.match(r'lbann_.*', entry.name):
                    return entry.path

def get_system_seq_launch(cluster):
    if cluster in ['lassen', 'ray']:
        return ['lrun', '-1', '--smpiargs=\"-disable_gpu_hooks\"']
    elif cluster in ['tioga', 'corona']:
        return ['flux mini run', '-N1', '-n1']
    return ['srun', '-N1', '-n1', '--mpibind=off']

def get_system_mpi_launch(cluster):
    if cluster in ['lassen', 'ray']:
        return ['jsrun', '-n2', '-r1', '-a4', '-c', 'ALL_CPUS', '-g', 'ALL_GPUS', '-d', 'packed', '-b', 'packed:10']
    elif cluster == 'pascal':
        return ['srun', '-N2', '--ntasks-per-node=2', '--mpibind=off']
    elif cluster in ['tioga', 'corona']:
        return ['flux mini run', '-N2', '-n2', '-g1', '-o gpu-affinity=per-task', '-o cpu-affinity=per-task']
    else: # Catalyst
        return ['srun', '-N2', '--ntasks-per-node=4']

# Notice that these tests will automatically skip if the executable
# doesn't exist. Since we do not save the testing executable as a
# GitLab CI artifact on Catalyst, Corona, or Pascal, this should only
# run on Ray and Lassen in GitLab CI testing pipelines.
def test_run_sequential_catch_tests(cluster, dirname):
    if cluster != 'lassen':
      message = f'{os.path.basename(__file__)} is only required on lassen due to limitations of CI testing'
      print('Skip - ' + message)
      pytest.skip(message)
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
    seq_catch_args = [seq_catch_exe, '-r', 'junit', '-o', seq_output_file]
    output = sp.run(seq_launch + seq_catch_args, cwd=build_dir)
    if output.returncode != 0:
        raise AssertionError('return_code={%d}' % output.returncode)

def test_run_parallel_catch_tests(cluster, dirname):
    if cluster != 'lassen':
      message = f'{os.path.basename(__file__)} is only required on lassen due to limitations of CI testing'
      print('Skip - ' + message)
      pytest.skip(message)
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
    mpi_catch_args = [mpi_catch_exe, '-r', 'junit', '-o', mpi_output_file]
    output = sp.run(mpi_launch + mpi_catch_args, cwd=build_dir)
    if output.returncode != 0:
        raise AssertionError('return_code={%d}' % output.returncode)

def test_run_parallel_filesystem_catch_tests(cluster, dirname):
    if cluster != 'lassen':
      message = f'{os.path.basename(__file__)} is only required on lassen due to limitations of CI testing'
      print('Skip - ' + message)
      pytest.skip(message)
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
    mpi_catch_args = [mpi_catch_exe, '"[filesystem]"', '-r', 'junit', '-o', mpi_output_file]
    output = sp.run(mpi_launch + mpi_catch_args, cwd=build_dir)
    if output.returncode != 0:
        raise AssertionError('return_code={%d}' % output.returncode)
