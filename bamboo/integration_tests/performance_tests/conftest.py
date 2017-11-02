import pytest

def pytest_addoption(parser):
    parser.addoption("--exe", action="store",
        help="--exe specifies the executable")
    parser.addoption("--dirname", action="store",
        help="--dirname specifies the top-level directory")

@pytest.fixture
def exe(request):
    return request.config.getoption("--exe")

@pytest.fixture
def dirname(request):
    return request.config.getoption("--dirname")
