import pytest
import os, re, subprocess

def test_clang(dirname):
    spack_skeleton(dirname, 'clang@4.0.0', 'mvapich2@2.2', True)
    build_skeleton(dirname, 'clang@4.0.0', 'mvapich2@2.2', True)

def test_gcc4(dirname):
    spack_skeleton(dirname, 'gcc@4.9.3', 'mvapich2@2.2', True)
    build_skeleton(dirname, 'gcc@4.9.3', 'mvapich2@2.2', True)

def test_gcc7(dirname):
    spack_skeleton(dirname, 'gcc@7.1.0', 'mvapich2@2.2', True)
    build_skeleton(dirname, 'gcc@7.1.0', 'mvapich2@2.2', True)

def test_intel(dirname):
    spack_skeleton(dirname, 'intel@18.0.0', 'mvapich2@2.2', True)
    build_skeleton(dirname, 'intel@18.0.0', 'mvapich2@2.2', True)

def spack_skeleton(dirname, compiler, mpi_lib, should_log):
    output_file_name = '%s/bamboo/compiler_tests/%s_spack_output.txt' % (dirname, re.sub('[@\.]', '_', compiler))
    os.chdir('%s/bamboo/compiler_tests/builds' % dirname)
    command = '%s/scripts/spack_recipes/build_lbann.sh -c %s -m %s > %s' % (dirname, compiler, mpi_lib, output_file_name)
    return_code = os.system(command)
    os.chdir('..')
    if should_log or (return_code != 0):
        output_file = open(output_file_name, 'r')
        for line in output_file:
            print('%s: %s' % (output_file_name, line))
    assert return_code == 0

def build_skeleton(dirname, compiler, mpi_lib, should_log):
    output_file_name = '%s/bamboo/compiler_tests/%s_build_output.txt' % (dirname, re.sub('[@\.]', '_', compiler))
    compiler = compiler.replace('@', '-')
    mpi_lib = mpi_lib.replace('@', '-')
    cluster = re.sub('[0-9]+\n', '', subprocess.check_output(['hostname']))
    os.chdir('%s/bamboo/compiler_tests/builds/%s_%s_x86_64_%s_openblas_rel/build' % (dirname, cluster, compiler, mpi_lib))
    command = 'make -j all > %s' % output_file_name
    return_code = os.system(command)
    os.chdir('../..')
    if should_log or (return_code != 0):
        output_file = open(output_file_name, 'r')
        for line in output_file:
            print('%s: %s' % (output_file_name, line))
    assert return_code == 0
