# import sys
# sys.path.insert(0, '../common_python')
# import tools
import pytest
import os, re, subprocess


def test_compiler_build_script(cluster, dirname):
    if cluster not in ['corona', 'lassen', 'pascal']:
        e = 'test_compiler_build_script: Unsupported Cluster %s' % cluster
        print('Skip - ' + e)
        pytest.skip(e)
    output_file_name = '%s/bamboo/compiler_tests/output/build_script_output.txt' % (dirname)
    error_file_name = '%s/bamboo/compiler_tests/error/build_script_error.txt' % (dirname)
    command = '%s/bamboo/compiler_tests/build_script.sh > %s 2> %s' % (
        dirname, output_file_name, error_file_name)
    return_code = os.system(command)
    if return_code != 0:
        output_file = open(output_file_name, 'r')
        for line in output_file:
            print('%s: %s' % (output_file_name, line))
        error_file = open(error_file_name, 'r')
        for line in error_file:
            print('%s: %s' % (error_file_name, line))
    assert return_code == 0


def test_compiler_clang6_release(cluster, dirname):
    skeleton_clang6(cluster, dirname, False)
    path = '%s/bamboo/compiler_tests/builds/%s_clang-6.0.0_rel/build/model_zoo/lbann' % (dirname, cluster)
    if not os.path.exists(path):
        path = '%s/build/clang.Release.%s.llnl.gov/install/bin/lbann' % (dirname, cluster)
        assert os.path.exists(path)


def test_compiler_clang6_debug(cluster, dirname):
    skeleton_clang6(cluster, dirname, True)
    path = '%s/bamboo/compiler_tests/builds/%s_clang-6.0.0_debug/build/model_zoo/lbann' % (dirname, cluster)
    if not os.path.exists(path):
        path = '%s/build/clang.Debug.%s.llnl.gov/install/bin/lbann' % (dirname, cluster)
        assert os.path.exists(path)


def test_compiler_gcc7_release(cluster, dirname):
    skeleton_gcc7(cluster, dirname, False)
    path = '%s/bamboo/compiler_tests/builds/%s_gcc-7.1.0_rel/build/model_zoo/lbann' % (dirname, cluster)
    if not os.path.exists(path):
        path = '%s/build/gnu.Release.%s.llnl.gov/install/bin/lbann' % (dirname, cluster)
        assert os.path.exists(path)


def test_compiler_gcc7_debug(cluster, dirname):
    skeleton_gcc7(cluster, dirname, True)
    path = '%s/bamboo/compiler_tests/builds/%s_gcc-7.1.0_debug/build/model_zoo/lbann' % (dirname, cluster)
    if not os.path.exists(path):
        path = '%s/build/gnu.Debug.%s.llnl.gov/install/bin/lbann' % (dirname, cluster)
        assert os.path.exists(path)


def test_compiler_intel19_release(cluster, dirname):
    skeleton_intel19(cluster, dirname, False)
    path = '%s/bamboo/compiler_tests/builds/%s_intel-19.0.0_rel/build/model_zoo/lbann' % (dirname, cluster)
    if not os.path.exists(path):
        path = '%s/build/intel.Release.%s.llnl.gov/install/bin/lbann' % (dirname, cluster)
        assert os.path.exists(path)


def test_compiler_intel19_debug(cluster, dirname):
    skeleton_intel19(cluster, dirname, True)
    path = '%s/bamboo/compiler_tests/builds/%s_intel-19.0.0_debug/build/model_zoo/lbann' % (dirname, cluster)
    if not os.path.exists(path):
        path = '%s/build/intel.Debug.%s.llnl.gov/install/bin/lbann' % (dirname, cluster)
        assert os.path.exists(path)


def skeleton_clang6(cluster, dir_name, debug, should_log=False):
    if cluster not in ['catalyst']:
        e = 'skeleton_clang6: Unsupported Cluster %s' % cluster
        print('Skip - ' + e)
        pytest.skip(e)
    try:
        spack_skeleton(dir_name, 'clang@6.0.0', 'mvapich2@2.2', debug,
                       should_log)
        build_skeleton(dir_name, 'clang@6.0.0', debug, should_log)
    except AssertionError as e:
        print(e)
        build_script(cluster, dir_name, 'clang6', debug)


def skeleton_gcc7(cluster, dir_name, debug, should_log=False):
    if cluster not in ['catalyst', 'pascal']:
        e = 'skeleton_gcc7: Unsupported Cluster %s' % cluster
        print('Skip - ' + e)
        pytest.skip(e)
    try:
        spack_skeleton(dir_name, 'gcc@7.1.0', 'mvapich2@2.2', debug, should_log)
        build_skeleton(dir_name, 'gcc@7.1.0', debug, should_log)
    except AssertionError as e:
        print(e)
        build_script(cluster, dir_name, 'gcc7', debug)


def skeleton_intel19(cluster, dir_name, debug, should_log=False):
    if cluster not in []:  # Taking out 'catalyst'
        e = 'skeleton_intel19: Unsupported Cluster %s' % cluster
        print('Skip - ' + e)
        pytest.skip(e)
    try:
        spack_skeleton(dir_name, 'intel@19.0.0', 'mvapich2@2.2', debug,
                       should_log)
        build_skeleton(dir_name, 'intel@19.0.0', debug, should_log)
    except AssertionError as e:
        print(e)
        build_script(cluster, dir_name, 'intel19', debug)


def spack_skeleton(dir_name, compiler, mpi_lib, debug, should_log):
    compiler_underscored = re.sub('[@\.]', '_', compiler)
    if debug:
        build_type = 'debug'
    else:
        build_type = 'rel'
    output_file_name = '%s/bamboo/compiler_tests/output/%s_%s_spack_output.txt' % (dir_name, compiler_underscored, build_type)
    error_file_name = '%s/bamboo/compiler_tests/error/%s_%s_spack_error.txt' % (dir_name, compiler_underscored, build_type)
    os.chdir('%s/bamboo/compiler_tests/builds' % dir_name)
    debug_flag = ''
    if debug:
        debug_flag = ' -d'
    command = '%s/scripts/spack_recipes/build_lbann.sh -c %s -m %s%s > %s 2> %s' % (
        dir_name, compiler, mpi_lib, debug_flag, output_file_name, error_file_name)
    return_code = os.system(command)
    os.chdir('..')
    if should_log or (return_code != 0):
        output_file = open(output_file_name, 'r')
        for line in output_file:
            print('%s: %s' % (output_file_name, line))
        error_file = open(error_file_name, 'r')
        for line in error_file:
            print('%s: %s' % (error_file_name, line))
    assert return_code == 0


def build_skeleton(dir_name, compiler, debug, should_log):
    compiler_underscored = re.sub('[@\.]', '_', compiler)
    if debug:
        build_type = 'debug'
    else:
        build_type = 'rel'
    output_file_name = '%s/bamboo/compiler_tests/output/%s_%s_build_output.txt' % (dir_name, compiler_underscored, build_type)
    error_file_name = '%s/bamboo/compiler_tests/error/%s_%s_build_error.txt' % (dir_name, compiler_underscored, build_type)
    compiler = compiler.replace('@', '-')
    #mpi_lib = mpi_lib.replace('@', '-')
    cluster = re.sub('[0-9]+', '', subprocess.check_output('hostname'.split()).decode('utf-8').strip())
    # For reference:
    # Commenting out for now. These additions to path name will likely return
    # one day, so I am not removing them entirely.
    # x86_64 <=> catalyst, pascal
    # ppc64le <=> ray
    #architecture = subprocess.check_output('uname -m'.split()).decode('utf-8').strip()
    #if cluster == 'ray':
    #    architecture += '_gpu_cuda-9.2.64_cudnn-7.0'
    #elif cluster == 'pascal':
    #    architecture += '_gpu_cuda-9.1.85_cudnn-7.1'
    os.chdir('%s/bamboo/compiler_tests/builds/%s_%s_%s/build' % (dir_name, cluster, compiler, build_type))
    command = 'make -j all > %s 2> %s' % (output_file_name, error_file_name)
    return_code = os.system(command)
    os.chdir('../..')
    if should_log or (return_code != 0):
        output_file = open(output_file_name, 'r')
        for line in output_file:
            print('%s: %s' % (output_file_name, line))
        error_file = open(error_file_name, 'r')
        for line in error_file:
            print('%s: %s' % (error_file_name, line))
    assert return_code == 0


def build_script(cluster, dirname, compiler, debug):
    print(('Running build_script for cluster={cluster},'
           ' compiler={compiler}, debug={debug}.').format(
        cluster=cluster, compiler=compiler, debug=debug))
    if debug:
        build = 'debug'
        debug_flag = '--debug'
    else:
        build = 'release'
        debug_flag = ''
    output_file_name = '%s/bamboo/compiler_tests/output/%s_%s_%s_build_script_output.txt' % (dirname, cluster, compiler, build)
    error_file_name = '%s/bamboo/compiler_tests/error/%s_%s_%s_build_script_error.txt' % (dirname, cluster, compiler, build)
    command = '%s/bamboo/compiler_tests/build_script_specific.sh --compiler %s %s> %s 2> %s' % (dirname, compiler, debug_flag, output_file_name, error_file_name)
    return_code = os.system(command)
    if return_code != 0:
        output_file = open(output_file_name, 'r')
        for line in output_file:
            print('%s: %s' % (output_file_name, line))
        error_file = open(error_file_name, 'r')
        for line in error_file:
            print('%s: %s' % (error_file_name, line))
    assert return_code == 0
