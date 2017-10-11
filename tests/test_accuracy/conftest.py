import pytest

def pytest_addoption(parser):
    parser.addoption("--log", action="store", default=0,
        help="--log=1 to keep trimmed accuracy files. Default (--log=0) removes files")

@pytest.fixture
def log(request):
return request.config.getoption("--log")
