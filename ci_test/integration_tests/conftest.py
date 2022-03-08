import sys
sys.path.insert(0, '../common_python')
import tools
import pytest, re, subprocess


def pytest_addoption(parser):
    cluster = re.sub('[0-9]+', '', subprocess.check_output(
        'hostname'.split()).decode('utf-8').strip())
    default_dirname = subprocess.check_output(
        'git rev-parse --show-toplevel'.split()).decode('utf-8').strip()

    parser.addoption('--cluster', action='store', default=cluster,
                     help='--cluster=<cluster> to specify the cluster being run on, for the purpose of determing which commands to use. Default the current cluster')
    parser.addoption('--dirname', action='store', default=default_dirname,
                     help='--dirname=<path_to_dir> to specify the top-level directory. Default directory of build_lbann_lc executable')
    parser.addoption('--weekly', action='store_true', default=False,
                     help='--weekly specifies that the test should ONLY be run weekly, not nightly. Default False')


@pytest.fixture
def cluster(request):
    return request.config.getoption('--cluster')


@pytest.fixture
def dirname(request):
    return request.config.getoption('--dirname')


@pytest.fixture
def weekly(request):
    return request.config.getoption('--weekly')
