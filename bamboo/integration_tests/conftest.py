import pytest, os, re, subprocess

def pytest_addoption(parser):
    cluster = re.sub('[0-9]+', '', subprocess.check_output('hostname'.split()).strip())
    default_dirname = subprocess.check_output('git rev-parse --show-toplevel'.split()).strip()
    key = 'bamboo_planKey'
    if key in os.environ:
        plan = os.environ['bamboo_planKey']
        default_exe = '%s/../%s-BDE/build/%s.llnl.gov/model_zoo/lbann' % (default_dirname, plan, cluster)
    else:
        default_exe = '%s/build/%s.llnl.gov/model_zoo/lbann' % (default_dirname,cluster)
    parser.addoption('--log', action='store', default=0,
                     help='--log=1 to keep trimmed accuracy files. Default (--log=0) removes files')
    parser.addoption('--exe', action='store', default=default_exe,
                     help='--exe=<path_to_lbann> to specify Lbann path. Default build_lbann_lc executable')
    parser.addoption('--dirname', action='store', default=default_dirname,
                     help='--dirname=<path_to_dir> to specify the top-level directory. Default directory of build_lbann_lc executable')
    parser.addoption('--weekly', action='store_true', default=False,
                     help='--weekly specifies that the test should ONLY be run weekly, not nightly')

@pytest.fixture
def log(request):
    return request.config.getoption('--log')

@pytest.fixture
def exe(request):
    return request.config.getoption('--exe')

@pytest.fixture
def dirname(request):
    return request.config.getoption('--dirname')

@pytest.fixture
def weekly(request):
    return request.config.getoption('--weekly')
