import pytest, re, subprocess

def pytest_addoption(parser):
    cluster = re.sub('[0-9]+', '', subprocess.check_output('hostname'.split()).strip())
    default_dirname = subprocess.check_output('git rev-parse --show-toplevel'.split()).strip()
    default_exe = '%s/../LBANN-NIGHTD-BDE/build/%s.llnl.gov/model_zoo/lbann' % (default_dirname, cluster)
    parser.addoption('--log', action='store', default=0,
                     help='--log=1 to keep trimmed accuracy files. Default (--log=0) removes files')
    parser.addoption('--exe', action='store', default=default_exe,
                     help='--exe=<path_to_lbann> to specify Lbann path. Default build_lbann_lc executable')
    parser.addoption('--dirname', action='store', default=default_dirname,
                     help='--dirname=<path_to_dir> to specify the top-level directory. Default directory of build_lbann_lc executable')

@pytest.fixture
def log(request):
    return request.config.getoption('--log')

@pytest.fixture
def exe(request):
    return request.config.getoption('--exe')

@pytest.fixture
def dirname(request):
    return request.config.getoption('--dirname')
