import sys
sys.path.insert(0, '../common_python')
import tools
import pytest, re, subprocess

def pytest_addoption(parser):
    cluster = re.sub('[0-9]+', '', subprocess.check_output(
        'hostname'.split()).strip())
    default_dirname = subprocess.check_output(
        'git rev-parse --show-toplevel'.split()).strip()
    default_exes = tools.get_default_exes(default_dirname, cluster)

    parser.addoption('--cluster', action='store', default=cluster,
                     help='--cluster=<cluster> to specify the cluster being run on, for the purpose of determing which commands to use. Default the current cluster')
    parser.addoption('--dirname', action='store', default=default_dirname,
                     help='--dirname specifies the top-level directory')
    parser.addoption('--exes', action='store', default=default_exes,
                     help='--exes={compiler_name: path}')
    # For local testing only
    parser.addoption('--exe', action='store', help='--exe=<hand-picked executable>')

@pytest.fixture
def cluster(request):
    return request.config.getoption('--cluster')

@pytest.fixture
def dirname(request):
    return request.config.getoption('--dirname')

@pytest.fixture
def exes(request):
    return request.config.getoption('--exes')

@pytest.fixture
def exe(request):
    return request.config.getoption('--exe')
