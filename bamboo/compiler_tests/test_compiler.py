import pytest
import os, re, subprocess

def test_compiler_build_script(cluster, dirname):
    output_file_name = '%s/bamboo/compiler_tests/output/build_script_output.txt' % (dirname)
    error_file_name = '%s/bamboo/compiler_tests/error/build_script_error.txt' % (dirname)
    command = '%s/bamboo/compiler_tests/build_script.sh > %s 2> %s' % (dirname, output_file_name, error_file_name)
    return_code = os.system(command)
    if return_code != 0:
        output_file = open(output_file_name, 'r')
        for line in output_file:
            print('%s: %s' % (output_file_name, line))
        error_file = open(error_file_name, 'r')
        for line in error_file:
            print('%s: %s' % (error_file_name, line))
    assert return_code == 0

def test_compiler_clang4_release(cluster, dirname):
    skeleton_clang4(cluster, dirname, False)

def test_compiler_clang4_debug(cluster, dirname):
    skeleton_clang4(cluster, dirname, True)

def test_compiler_gcc4_release(cluster, dirname):
    skeleton_gcc4(cluster, dirname, False)

def test_compiler_gcc4_debug(cluster, dirname):
    skeleton_gcc4(cluster, dirname, True)

def test_compiler_gcc7_release(cluster, dirname):
    skeleton_gcc7(cluster, dirname, False)

def test_compiler_gcc7_debug(cluster, dirname):
    skeleton_gcc7(cluster, dirname, True)

def test_compiler_intel18_release(cluster, dirname):
    skeleton_intel18(cluster, dirname, False)

def test_compiler_intel18_debug(cluster, dirname):
    skeleton_intel18(cluster, dirname, True)

def skeleton_clang4(cluster, dir_name, debug, should_log=False):
    if cluster in ['catalyst', 'pascal', 'quartz']:
        spack_skeleton(dir_name, 'clang@4.0.0', 'mvapich2@2.2', debug, should_log)
        build_skeleton(dir_name, 'clang@4.0.0', 'mvapich2@2.2', debug, should_log)
    else:
        pytest.skip('Unsupported Cluster %s' % cluster)

def skeleton_gcc4(cluster, dir_name, debug, should_log=False):
    if cluster in ['catalyst', 'pascal', 'quartz', 'ray']:
        if cluster == 'catalyst':
            mpi = 'mvapich2@2.2'
        elif cluster == 'ray':
            mpi = 'spectrum-mpi@10.1.0'
        else:
            raise Exception('Unsupported Cluster %s' % cluster)
        spack_skeleton(dir_name, 'gcc@4.9.3', mpi, debug, should_log)
        build_skeleton(dir_name, 'gcc@4.9.3', mpi, debug, should_log)
    else:
        pytest.skip('Unsupported Cluster %s' % cluster)

def skeleton_gcc7(cluster, dir_name, debug, should_log=False):
    if cluster in ['catalyst', 'pascal', 'quartz']:
        spack_skeleton(dir_name, 'gcc@7.1.0', 'mvapich2@2.2', debug, should_log)
        build_skeleton(dir_name, 'gcc@7.1.0', 'mvapich2@2.2', debug, should_log)
    else:
        pytest.skip('Unsupported Cluster %s' % cluster)

def skeleton_intel18(cluster, dir_name, debug, should_log=False):
    if cluster in ['catalyst', 'pascal', 'quartz']:
        spack_skeleton(dir_name, 'intel@18.0.0', 'mvapich2@2.2', debug, should_log)
        build_skeleton(dir_name, 'intel@18.0.0', 'mvapich2@2.2', debug, should_log)
    else:
        pytest.skip('Unsupported Cluster %s' % cluster)

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

def build_skeleton(dir_name, compiler, mpi_lib, debug, should_log):
    compiler_underscored = re.sub('[@\.]', '_', compiler)
    if debug:
        build_type = 'debug'
    else:
        build_type = 'rel'
    output_file_name = '%s/bamboo/compiler_tests/output/%s_%s_build_output.txt' % (dir_name, compiler_underscored, build_type)
    error_file_name = '%s/bamboo/compiler_tests/error/%s_%s_build_error.txt' % (dir_name, compiler_underscored, build_type)
    compiler = compiler.replace('@', '-')
    mpi_lib = mpi_lib.replace('@', '-')
    cluster = re.sub('[0-9]+\n', '', subprocess.check_output(['hostname']))
    if cluster in ['catalyst', 'surface']:
        architecture = 'x86_64'
    elif cluster == 'ray':
        architecture = 'ppc64le_gpu'
    else:
        raise Exception('Unsupported Cluster %s' % cluster)
    os.chdir('%s/bamboo/compiler_tests/builds/%s_%s_%s_%s_openblas_%s/build' % (dir_name, cluster, compiler, architecture, mpi_lib, build_type))
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
