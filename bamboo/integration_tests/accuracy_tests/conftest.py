import pytest

def pytest_addoption(parser):
    parser.addoption("--log", action="store", default=0,
        help="--log=1 to keep trimmed accuracy files. Default (--log=0) removes files")
    parser.addoption("--exe", action="store", default='',
        help="--exe=<path_to_lbann> to specify Lbann path. Default (--exe='') uses build_lbann_lc exe")

@pytest.fixture
def log(request):
    return request.config.getoption("--log")

@pytest.fixture
def exe(request):
    return request.config.getoption("--exe")
