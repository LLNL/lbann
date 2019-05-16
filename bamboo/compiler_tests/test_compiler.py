import sys
sys.path.insert(0, '../common_python')
import tools
import pytest
import os, re, subprocess


def test_compiler_build_script(cluster, dirname):
    if cluster not in []:
        e = 'test_compiler_build_script: Unsupported Cluster %s' % cluster
        print('Skip - ' + e)
        pytest.skip(e)
    output_file_name = '%s/bamboo/compiler_tests/output/build_script_output.txt' % (dirname)
    error_file_name = '%s/bamboo/compiler_tests/error/build_script_error.txt' % (dirname)
    command = '%s/bamboo/compiler_tests/build_script.sh > %s 2> %s' % (
        dirname, output_file_name, error_file_name)
    return_code = os.system(command)
    tools.assert_success(return_code, error_file_name)


def test_compiler_clang6_release(cluster, dirname):
    skeleton_clang6(cluster, dirname, False)
    path = '%s/bamboo/compiler_tests/builds/%s_clang-6.0.0_rel/build/model_zoo/lbann' % (dirname, cluster)
    assert os.path.exists(path)


def test_compiler_clang6_debug(cluster, dirname):
    skeleton_clang6(cluster, dirname, True)
    path = '%s/bamboo/compiler_tests/builds/%s_clang-6.0.0_debug/build/model_zoo/lbann' % (dirname, cluster)
    assert os.path.exists(path)


def test_compiler_gcc7_release(cluster, dirname):
    skeleton_gcc7(cluster, dirname, False)
    path = '%s/bamboo/compiler_tests/builds/%s_gcc-7.1.0_rel/build/model_zoo/lbann' % (dirname, cluster)
    assert os.path.exists(path)


def test_compiler_gcc7_debug(cluster, dirname):
    skeleton_gcc7(cluster, dirname, True)
    path = '%s/bamboo/compiler_tests/builds/%s_gcc-7.1.0_debug/build/model_zoo/lbann' % (dirname, cluster)
    assert os.path.exists(path)


def test_compiler_intel19_release(cluster, dirname):
    skeleton_intel19(cluster, dirname, False)
    path = '%s/bamboo/compiler_tests/builds/%s_intel-19.0.0_rel/build/model_zoo/lbann' % (dirname, cluster)
    assert os.path.exists(path)


def test_compiler_intel19_debug(cluster, dirname):
    skeleton_intel19(cluster, dirname, True)
    path = '%s/bamboo/compiler_tests/builds/%s_intel-19.0.0_debug/build/model_zoo/lbann' % (dirname, cluster)
    assert os.path.exists(path)


def skeleton_clang6(cluster, dir_name, debug):
    if cluster in ['catalyst']:
        compile(dir_name, 'clang@6.0.0', debug, 'mvapich2@2.2')
    else:
        e = 'skeleton_clang6: Unsupported Cluster %s' % cluster
        print('Skip - ' + e)
        pytest.skip(e)


def skeleton_gcc7(cluster, dir_name, debug):
    if cluster in ['catalyst', 'corona', 'lassen', 'pascal']:
        compile(dir_name, 'gcc@7.1.0', debug, 'mvapich2@2.2')
    else:
        e = 'skeleton_gcc7: Unsupported Cluster %s' % cluster
        print('Skip - ' + e)
        pytest.skip(e)


def skeleton_intel19(cluster, dir_name, debug):
    if cluster in ['catalyst']:
        compile(dir_name, 'intel@19.0.0', debug, 'mvapich2@2.2')
    else:
        e = 'skeleton_intel19: Unsupported Cluster %s' % cluster
        print('Skip - ' + e)
        pytest.skip(e)


def compile(dir_name, compiler, debug, mpi):
    cluster = re.sub('[0-9]+', '',
                     subprocess.check_output('hostname'.split()).decode(
                         'utf-8').strip())
    if cluster in ['catalyst', 'corona', 'pascal']:
        architecture = 'x86_64_cuda'
    elif cluster in ['lassen']:
        architecture = 'ppc64le_cuda'
    else:
        raise Exception('Invalid cluster={c}'.format(c=cluster))
    if debug:
        debug_command = ' +debug'
        debug_string = '_debug'
    else:
        debug_command = ''
        debug_string = ''
    compiler_string = compiler.replace('@', '_')
    compiler_string = compiler_string.replace('.', '_')

    spack = os.environ['bamboo_SPACK']

    # https://lbann.readthedocs.io/en/latest/building_lbann.html#building-installing-lbann-as-a-user
    # cd <path to LBANN repo>/spack_environments/users/llnl_lc/<arch>_gpu/
    os.chdir('{lbann_path}/spack_environments/users/llnl_lc/{arch}'.format(
        lbann_path=dir_name, arch=architecture))

    spack_command = '{s} compiler find; {s} install lbann@develop %{c}{d} +gpu +nccl ^{m} ^hydrogen@develop ^aluminum@master'.format(
        s=spack, c=compiler, d=debug_command, m=mpi)
    print('spack_command={s}'.format(s=spack_command))
    output_file_name = '{d}/bamboo/compiler_tests/output/{cl}_{cs}{ds}_spack_output.txt'.format(
        d=dir_name, cl=cluster, cs=compiler_string, ds=debug_string)
    error_file_name = '{d}/bamboo/compiler_tests/error/{cl}_{cs}{ds}_spack_error.txt'.format(
        d=dir_name, cl=cluster, cs=compiler_string, ds=debug_string)
    spack_command += '> {o} 2> {e}'.format(
        o=output_file_name, e=error_file_name)
    return_code = os.system(spack_command)
    tools.assert_success(return_code, error_file_name)

    build_command = 'ml load lbann'
    output_file_name = '{d}/bamboo/compiler_tests/output/{cl}_{cs}{ds}_build_output.txt'.format(
        d=dir_name, cl=cluster, cs=compiler_string, ds=debug_string)
    error_file_name = '{d}/bamboo/compiler_tests/error/{cl}_{cs}{ds}_build_error.txt'.format(
        d=dir_name, cl=cluster, cs=compiler_string, ds=debug_string)
    build_command += '> {o} 2> {e}'.format(
        o=output_file_name, e=error_file_name)
    return_code = os.system(build_command)
    tools.assert_success(return_code, error_file_name)
