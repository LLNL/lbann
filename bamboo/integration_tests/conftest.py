import sys
sys.path.insert(0, '../common_python')
import tools
import pytest, re, subprocess


def pytest_addoption(parser):
    cluster = re.sub('[0-9]+', '', subprocess.check_output(
        'hostname'.split()).decode('utf-8').strip())
    default_dirname = subprocess.check_output(
        'git rev-parse --show-toplevel'.split()).decode('utf-8').strip()
    default_exes = tools.get_default_exes(default_dirname, cluster)

    parser.addoption('--cluster', action='store', default=cluster,
                     help='--cluster=<cluster> to specify the cluster being run on, for the purpose of determing which commands to use. Default the current cluster')
    parser.addoption('--debug_build', action='store_true', default=False,
                      help='--debug_build specifies that debug tests should be run, even without doing a --weekly build. Default False')
    parser.addoption('--dirname', action='store', default=default_dirname,
                     help='--dirname=<path_to_dir> to specify the top-level directory. Default directory of build_lbann_lc executable')
    parser.addoption('--exes', action='store', default=default_exes,
                     help='--exes={compiler_name: path}')
    parser.addoption('--run', action='store_true', default=False,
                     help='--run specifies that a test normally ignored should be run. Default False')
    parser.addoption('--weekly', action='store_true', default=False,
                     help='--weekly specifies that the test should ONLY be run weekly, not nightly. Default False')
    # For local testing only
    parser.addoption('--exe', action='store', help='--exe=<hand-picked executable>')


@pytest.fixture
def cluster(request):
    return request.config.getoption('--cluster')


@pytest.fixture
def debug_build(request):
    return request.config.getoption('--debug_build')


@pytest.fixture
def dirname(request):
    return request.config.getoption('--dirname')


@pytest.fixture
def exes(request):
    return request.config.getoption('--exes')


@pytest.fixture
def run(request):
    return request.config.getoption('--run')


@pytest.fixture
def weekly(request):
    return request.config.getoption('--weekly')


@pytest.fixture
def exe(request):
    return request.config.getoption('--exe')
