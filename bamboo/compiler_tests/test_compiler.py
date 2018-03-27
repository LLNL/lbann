import pytest
import os, re, subprocess

def test_compiler_clang(cluster, dirname):
    if cluster == 'catalyst':
        spack_skeleton(dirname, 'clang@4.0.0', 'mvapich2@2.2', False)
        build_skeleton(dirname, 'clang@4.0.0', 'mvapich2@2.2', False)
    else:
        pytest.skip('Unsupported Cluster %s' % cluster)

def test_compiler_gcc4(cluster, dirname):
    if cluster in ['catalyst', 'ray']:
        if cluster == 'catalyst':
            mpi = 'mvapich2@2.2'
        elif cluster == 'ray':
            mpi = 'spectrum-mpi@10.1.0'
        else:
            raise Exception('Unsupported Cluster %s' % cluster)
        spack_skeleton(dirname, 'gcc@4.9.3', mpi, False)
        build_skeleton(dirname, 'gcc@4.9.3', mpi, False)
    else:
        pytest.skip('Unsupported Cluster %s' % cluster)

def test_compiler_gcc7(cluster, dirname):
    if cluster == 'catalyst':
        spack_skeleton(dirname, 'gcc@7.1.0', 'mvapich2@2.2', False)
        build_skeleton(dirname, 'gcc@7.1.0', 'mvapich2@2.2', False)
    else:
        pytest.skip('Unsupported Cluster %s' % cluster)

def test_compiler_intel(cluster, dirname):
    if cluster == 'catalyst':
        spack_skeleton(dirname, 'intel@18.0.0', 'mvapich2@2.2', False)
        build_skeleton(dirname, 'intel@18.0.0', 'mvapich2@2.2', False)
    else:
        pytest.skip('Unsupported Cluster %s' % cluster)

def spack_skeleton(dirname, compiler, mpi_lib, should_log, debug=False):
    output_file_name = '%s/bamboo/compiler_tests/output/%s_spack_output.txt' % (dirname, re.sub('[@\.]', '_', compiler))
    os.chdir('%s/bamboo/compiler_tests/builds' % dirname)
    command = '%s/scripts/spack_recipes/build_lbann.sh -c %s -m %s' % (dirname, compiler, mpi_lib)
    if debug:
        command += ' -d > %s' % output_file_name
    else:
        command += ' > %s' % output_file_name
    return_code = os.system(command)
    os.chdir('..')
    if should_log or (return_code != 0):
        output_file = open(output_file_name, 'r')
        for line in output_file:
            print('%s: %s' % (output_file_name, line))
    assert return_code == 0

def build_skeleton(dirname, compiler, mpi_lib, should_log, debug=False):
    output_file_name = '%s/bamboo/compiler_tests/output/%s_build_output.txt' % (dirname, re.sub('[@\.]', '_', compiler))
    compiler = compiler.replace('@', '-')
    mpi_lib = mpi_lib.replace('@', '-')
    cluster = re.sub('[0-9]+\n', '', subprocess.check_output(['hostname']))
    if debug:
        build_type = 'debug'
    else:
        build_type = 'rel'
    if cluster in ['catalyst', 'surface']:
        architecture = 'x86_64'
    elif cluster == 'ray':
        architecture = 'ppc64le_gpu'
    else:
        raise Exception('Unsupported Cluster %s' % cluster)
    os.chdir('%s/bamboo/compiler_tests/builds/%s_%s_%s_%s_openblas_%s/build' % (dirname, cluster, compiler, architecture, mpi_lib, build_type))
    command = 'make -j all > %s' % output_file_name
    return_code = os.system(command)
    os.chdir('../..')
    if should_log or (return_code != 0):
        output_file = open(output_file_name, 'r')
        for line in output_file:
            print('%s: %s' % (output_file_name, line))
    assert return_code == 0
