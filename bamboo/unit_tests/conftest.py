import pytest, os, re, subprocess

def pytest_addoption(parser):
    cluster = re.sub('[0-9]+', '', subprocess.check_output('hostname'.split()).strip())
    default_dirname = subprocess.check_output('git rev-parse --show-toplevel'.split()).strip()
    default_exe = '%s/build/gnu.%s.llnl.gov/lbann/build/model_zoo/lbann' % (default_dirname, cluster)
    parser.addoption('--cluster', action='store', default=cluster,
                     help='--cluster=<cluster> to specify the cluster being run on, for the purpose of determing which commands to use. Default the current cluster')
    parser.addoption('--dirname', action='store', default=default_dirname,
                     help='--dirname specifies the top-level directory')
    parser.addoption('--exe', action='store', default=default_exe,
                     help='--exe specifies the executable')
    
@pytest.fixture
def cluster(request):
    return request.config.getoption('--cluster')

@pytest.fixture
def dirname(request):
    return request.config.getoption('--dirname')

@pytest.fixture
def exe(request):
    return request.config.getoption('--exe')
