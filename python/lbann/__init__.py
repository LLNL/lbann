"""LBANN Python frontend."""
import sys
import os.path
import configparser

# Check for Python 3
if sys.version_info[0] != 3:
    raise ImportError('Python 3 is required')

# Try getting build-specific paths from config file
_config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'python_config.ini')
_lbann_exe = None
if os.path.isfile(_config_file):
    try:
        _config = configparser.ConfigParser()
        _config.read(_config_file)
        sys.path.append(os.path.dirname(_config['Paths']['lbann_pb2.py']))
        _lbann_exe = _config['Paths']['lbann_exe']
    except:
        pass
import lbann_pb2, callbacks_pb2
def lbann_exe():
    """LBANN executable."""
    return _lbann_exe if _lbann_exe else 'lbann'

# Import core functionality into lbann namespace
from lbann.callback import *
from lbann.layer import *
from lbann.metric import *
from lbann.model import *
from lbann.objective_function import *
from lbann.optimizer import *
from lbann.weights import *
from lbann.launcher import run
