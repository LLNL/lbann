# import sys
# sys.path.insert(0, '../common_python')
# import tools
import pytest
import os, re, subprocess


def test_compiler_build_script(cluster, dirname):
    if cluster in ['pascal']:
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
    else:
        e = 'test_compiler_build_script: Unsupported Cluster %s' % cluster
        print('Skip - ' + e)
        pytest.skip(e)


def test_compiler_clang4_release(cluster, dirname):
    try:
        skeleton_clang4(cluster, dirname, False)
    except AssertionError as e:
        print(e)
        build_script(cluster, dirname, 'clang4', False)
    path = '%s/bamboo/compiler_tests/builds/%s_clang-4.0.0_rel/build/model_zoo/lbann' % (dirname, cluster)
    if not os.path.exists(path):
        path = '%s/build/clang.Release.%s.llnl.gov/install/bin/lbann' % (dirname, cluster)
        assert os.path.exists(path)


def test_compiler_clang4_debug(cluster, dirname):
    try:
        skeleton_clang4(cluster, dirname, True)
    except AssertionError as e:
        print(e)
        build_script(cluster, dirname, 'clang4', True)
    path = '%s/bamboo/compiler_tests/builds/%s_clang-4.0.0_debug/build/model_zoo/lbann' % (dirname, cluster)
    if not os.path.exists(path):
        path = '%s/build/clang.Debug.%s.llnl.gov/install/bin/lbann' % (dirname, cluster)
        assert os.path.exists(path)


def test_compiler_gcc4_release(cluster, dirname):
    try:
        skeleton_gcc4(cluster, dirname, False)
    except AssertionError as e:
        print(e)
        build_script(cluster, dirname, 'gcc4', False)
    path = '%s/bamboo/compiler_tests/builds/%s_gcc-4.9.3_rel/build/model_zoo/lbann' % (dirname, cluster)
    assert os.path.exists(path)


def test_compiler_gcc4_debug(cluster, dirname):
    try:
        skeleton_gcc4(cluster, dirname, True)
    except AssertionError as e:
        print(e)
        build_script(cluster, dirname, 'gcc4', True)
    path = '%s/bamboo/compiler_tests/builds/%s_gcc-4.9.3_debug/build/model_zoo/lbann' % (dirname, cluster)
    assert os.path.exists(path)


def test_compiler_gcc7_release(cluster, dirname):
    try:
        skeleton_gcc7(cluster, dirname, False)
    except AssertionError as e:
        print(e)
        build_script(cluster, dirname, 'gcc7', False)
    path = '%s/bamboo/compiler_tests/builds/%s_gcc-7.1.0_rel/build/model_zoo/lbann' % (dirname, cluster)
    if not os.path.exists(path):
        path = '%s/build/gnu.Release.%s.llnl.gov/install/bin/lbann' % (dirname, cluster)
        assert os.path.exists(path)


def test_compiler_gcc7_debug(cluster, dirname):
    try:
        skeleton_gcc7(cluster, dirname, True)
    except AssertionError as e:
        print(e)
        build_script(cluster, dirname, 'gcc7', True)
    path = '%s/bamboo/compiler_tests/builds/%s_gcc-7.1.0_debug/build/model_zoo/lbann' % (dirname, cluster)
    if not os.path.exists(path):
        path = '%s/build/gnu.Debug.%s.llnl.gov/install/bin/lbann' % (dirname, cluster)
        assert os.path.exists(path)


def test_compiler_intel18_release(cluster, dirname):
    try:
        skeleton_intel18(cluster, dirname, False)
    except AssertionError as e:
        print(e)
        build_script(cluster, dirname, 'intel18', False)
    path = '%s/bamboo/compiler_tests/builds/%s_intel-18.0.0_rel/build/model_zoo/lbann' % (dirname, cluster)
    if not os.path.exists(path):
        path = '%s/build/intel.Release.%s.llnl.gov/install/bin/lbann' % (dirname, cluster)
        assert os.path.exists(path)


def test_compiler_intel18_debug(cluster, dirname):
    try:
        skeleton_intel18(cluster, dirname, True)
    except AssertionError as e:
        print(e)
        build_script(cluster, dirname, 'intel18', True)
    path = '%s/bamboo/compiler_tests/builds/%s_intel-18.0.0_debug/build/model_zoo/lbann' % (dirname, cluster)
    if not os.path.exists(path):
        path = '%s/build/intel.Debug.%s.llnl.gov/install/bin/lbann' % (dirname, cluster)
        assert os.path.exists(path)


def skeleton_clang4(cluster, dir_name, debug, should_log=False):
    if cluster in ['catalyst', 'quartz']:
        spack_skeleton(dir_name, 'clang@4.0.0', 'mvapich2@2.2', debug, should_log)
        build_skeleton(dir_name, 'clang@4.0.0', debug, should_log)
    else:
        e = 'skeleton_clang4: Unsupported Cluster %s' % cluster
        print('Skip - ' + e)
        pytest.skip(e)


def skeleton_gcc4(cluster, dir_name, debug, should_log=False):
    if cluster in ['quartz']:  # Taking out 'catalyst'
        mpi = 'mvapich2@2.2'
    elif cluster in ['surface']:  # Taking out 'pascal'
        mpi = 'mvapich2@2.2+cuda'
    elif cluster == 'ray':
        mpi = 'spectrum-mpi@2018.04.27'
    else:
        e = 'skeleton_gcc4: Unsupported Cluster %s' % cluster
        print('Skip - ' + e)
        pytest.skip(e)
    spack_skeleton(dir_name, 'gcc@4.9.3', mpi, debug, should_log)
    build_skeleton(dir_name, 'gcc@4.9.3', debug, should_log)


def skeleton_gcc7(cluster, dir_name, debug, should_log=False):
    if cluster in ['catalyst', 'quartz']:
        spack_skeleton(dir_name, 'gcc@7.1.0', 'mvapich2@2.2', debug, should_log)
        build_skeleton(dir_name, 'gcc@7.1.0', debug, should_log)
    else:
        e = 'skeleton_gcc7: Unsupported Cluster %s' % cluster
        print('Skip - ' + e)
        pytest.skip(e)


def skeleton_intel18(cluster, dir_name, debug, should_log=False):
    if cluster in ['quartz']:  # Taking out 'catalyst'
        spack_skeleton(dir_name, 'intel@18.0.0', 'mvapich2@2.2', debug, should_log)
        build_skeleton(dir_name, 'intel@18.0.0', debug, should_log)
    else:
        e = 'skeleton_intel18: Unsupported Cluster %s' % cluster
        print('Skip - ' + e)
        pytest.skip(e)


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
    cluster = re.sub('[0-9]+', '', subprocess.check_output('hostname'.split()).strip())
    # For reference:
    # Commenting out for now. These additions to path name will likely return
    # one day, so I am not removing them entirely.
    # x86_64 <=> catalyst, pascal, quartz, surface
    # ppc64le <=> ray
    #architecture = subprocess.check_output('uname -m'.split()).strip()
    #if cluster == 'ray':
    #    architecture += '_gpu_cuda-9.2.64_cudnn-7.0'
    #elif cluster == 'pascal':
    #    architecture += '_gpu_cuda-9.1.85_cudnn-7.1'
    #elif cluster == 'surface':
    #    architecture += '_gpu'
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
