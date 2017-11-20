import os, pytest, subprocess

def test_clang(dirname):
    spack_skeleton(dirname, 'clang@4.0.0', 'mvapich2@2.2')
    build_skeleton(dirname, 'clang@4.0.0', 'mvapich2@2.2')

def test_gcc4(dirname):
    spack_skeleton(dirname, 'gcc@4.9.3', 'mvapich2@2.2')
    build_skeleton(dirname, 'gcc@4.9.3', 'mvapich2@2.2')

def test_gcc7(dirname):
    spack_skeleton(dirname, 'gcc@7.1.0', 'mvapich2@2.2')
    build_skeleton(dirname, 'gcc@7.1.0', 'mvapich2@2.2')

def test_intel(dirname):
    spack_skeleton(dirname, 'intel@18.0.0', 'mvapich2@2.2')
    build_skeleton(dirname, 'intel@18.0.0', 'mvapich2@2.2')

def spack_skeleton(dirname, compiler, mpi_lib):
    os.chdir('%s/bamboo/compiler_tests/builds' % dirname)
    command = '%s/scripts/spack_recipes/build_lbann.sh -c %s -m %s' % (dirname, compiler, mpi_lib)
    return_code = subprocess.call(command.split())
    os.chdir('..')
    assert return_code == 0

def build_skeleton(dirname, compiler, mpi_lib):
    compiler = compiler.replace('@', '-')
    mpi_lib = mpi_lib.replace('@', '-')
    os.chdir('%s/bamboo/compiler_tests/builds/catalyst_%s_x86_64_%s_openblas_rel/build' % (dirname, compiler, mpi_lib))
    command = 'make -j all'
    return_code = subprocess.call(command.split())
    os.chdir('../..')
    assert return_code == 0
