import pytest, subprocess

def pytest_addoption(parser):
    default_dirname = subprocess.check_output('git rev-parse --show-toplevel'.split()).strip()
    parser.addoption('--dirname', action='store', default=default_dirname,
                     help='--dirname specifies the top-level directory')

@pytest.fixture
def dirname(request):
    return request.config.getoption('--dirname')
