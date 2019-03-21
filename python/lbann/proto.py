"""Generate LBANN experiment prototext files."""

import google.protobuf.text_format
import google.protobuf.message
from lbann import lbann_pb2

# TODO: remove these imports or move to __init__.py
from lbann.callback import *
from lbann.layer import *
from lbann.metric import *
from lbann.model import *
from lbann.objective_function import *
from lbann.optimizer import *
from lbann.weights import *

def save_prototext(filename, **kwargs):
    """Save a prototext file.

    LbannPB fields (e.g. `model`, `data_reader`, `optimizer`) are
    accepted via `kwargs`.

    """

    # Construct protobuf message
    for key, value in kwargs.items():
        if not isinstance(value, google.protobuf.message.Message):
            kwargs[key] = value.export_proto()
    pb = lbann_pb2.LbannPB(**kwargs)

    # Write to file
    with open(filename, 'wb') as f:
        f.write(google.protobuf.text_format.MessageToString(
            pb, use_index_order=True).encode())
