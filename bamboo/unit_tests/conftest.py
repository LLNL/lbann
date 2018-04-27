import pytest, os, re, subprocess

def pytest_addoption(parser):
    cluster = re.sub('[0-9]+', '', subprocess.check_output('hostname'.split()).strip())
    default_dirname = subprocess.check_output('git rev-parse --show-toplevel'.split()).strip()
    default_exes = {}
    default_exes['default'] = '%s/build/gnu.%s.llnl.gov/lbann/build/model_zoo/lbann' % (default_dirname, cluster)
    if cluster == 'catalyst':
        default_exes['clang4'] = '%s/bamboo/compiler_tests/builds/%s_clang-4.0.0_x86_64_mvapich2-2.2_openblas_rel/build/model_zoo/lbann' % (default_dirname, cluster)
        default_exes['gcc4'] = '%s/bamboo/compiler_tests/builds/%s_gcc-4.9.3_x86_64_mvapich2-2.2_openblas_rel/build/model_zoo/lbann' % (default_dirname, cluster)
        default_exes['gcc7'] = '%s/bamboo/compiler_tests/builds/%s_gcc-7.1.0_x86_64_mvapich2-2.2_openblas_rel/build/model_zoo/lbann' % (default_dirname, cluster)
        default_exes['intel18'] = '%s/bamboo/compiler_tests/builds/%s_intel-18.0.0_x86_64_mvapich2-2.2_openblas_rel/build/model_zoo/lbann' % (default_dirname, cluster)

    if cluster == 'ray':
        default_exes['gcc4'] = '%s/bamboo/compiler_tests/builds/%s_gcc-4.9.3_ppc64le_gpu_spectrum-mpi-10.1.0_openblas_rel/build/model_zoo/lbann' % (default_dirname, cluster)

    if cluster == 'surface':
        default_exes['gcc4'] = default_exes['default']

    parser.addoption('--cluster', action='store', default=cluster,
                     help='--cluster=<cluster> to specify the cluster being run on, for the purpose of determing which commands to use. Default the current cluster')
    parser.addoption('--dirname', action='store', default=default_dirname,
                     help='--dirname specifies the top-level directory')
    parser.addoption('--exes', action='store', default=default_exes,
                     help='--exes={compiler_name: path}')
    
@pytest.fixture
def cluster(request):
    return request.config.getoption('--cluster')

@pytest.fixture
def dirname(request):
    return request.config.getoption('--dirname')

@pytest.fixture
def exes(request):
    return request.config.getoption('--exes')
