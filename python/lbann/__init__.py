"""LBANN Python frontend."""
import sys
import os.path
import configparser
import warnings

# Check for Python 3
if sys.version_info[0] != 3:
    raise ImportError('Python 3 is required')

# Try getting build-specific paths from config file
if 'LBANN_PYTHON_CONFIG_FILE' in os.environ:
    _config_file = os.environ['LBANN_PYTHON_CONFIG_FILE']
else:
    _config_file = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'python_config.ini',
    )
_lbann_exe = None
_lbann_has_proto_definitions = False
if os.path.isfile(_config_file):
    try:
        _config = configparser.ConfigParser()
        _config.read(_config_file)
        sys.path.append(os.path.dirname(_config['Paths']['lbann_pb2.py']))
        _lbann_exe = _config['Paths']['lbann_exe']
        _lbann_has_proto_definitions = True
    except:
        pass

# Import Protobuf modules
if _lbann_has_proto_definitions:
    import lbann_pb2, callbacks_pb2, datatype_pb2, layers_pb2, metrics_pb2, model_pb2, objective_functions_pb2, operators_pb2, optimizers_pb2, reader_pb2, weights_pb2, trainer_pb2, training_algorithm_pb2
    protobuf_modules = (
        lbann_pb2,
        callbacks_pb2,
        datatype_pb2,
        layers_pb2,
        metrics_pb2,
        model_pb2,
        objective_functions_pb2,
        operators_pb2,
        optimizers_pb2,
        reader_pb2,
        weights_pb2,
        trainer_pb2,
        training_algorithm_pb2,
    )
    for module in protobuf_modules:
        for enum_name, enum_desc in module.DESCRIPTOR.enum_types_by_name.items():
            enum_val_to_num = {}
            enum_val_descs = enum_desc.values_by_name
            for val_name, val_desc in enum_val_descs.items():
                enum_val_to_num[val_name] = val_desc.number
            globals()[enum_name] = type(enum_name, (), enum_val_to_num)
else:
    lbann_pb2 = None
    callbacks_pb2 = None
    datatype_pb2 = None
    layers_pb2 = None
    metrics_pb2 = None
    model_pb2 = None
    objective_functions_pb2 = None
    operators_pb2 = None
    optimizers_pb2 = None
    reader_pb2 = None
    weights_pb2 = None
    trainer_pb2 = None
    training_algorithm_pb2 = None
    NoOptimizer = None
    warnings.warn('THE LBANN MODULE DOES NOT HAVE PROTOBUF DEFINITIONS. THIS MODE IS NOT FUNCTIONAL FOR MOST USERS AND SHOULD ONLY BE USED TO GENERATE DOCUMENTATION.')

def lbann_exe():
    """LBANN executable."""
    return _lbann_exe if _lbann_exe else 'lbann'

# Import core functionality into lbann namespace
from lbann.core.callback import *
from lbann.core.layer import *
from lbann.core.metric import *
from lbann.core.model import *
from lbann.core.objective_function import *
import lbann.core.operators as ops
from lbann.core.operator_layers import *
from lbann.core.optimizer import *
from lbann.core.trainer import *
from lbann.core.training_algorithm import *
from lbann.core.weights import *
from lbann.launcher import run
from lbann.lbann_features import *
from lbann.core.evaluate import evaluate
