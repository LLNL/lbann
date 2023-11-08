"""Generate LBANN experiment prototext files."""

import google.protobuf.text_format
import google.protobuf.message
from lbann import lbann_pb2, NoOptimizer


def save_prototext(filename, binary=False, **kwargs):
    """Save a prototext file.

    LbannPB fields (e.g. `model`, `data_reader`, `optimizer`) are
    accepted via `kwargs`. The `binary` field saves the experiment file as
    a protobuf binary message, rather than as a text format.

    """

    # Construct protobuf message
    message = lbann_pb2.LbannPB()
    field_names = message.DESCRIPTOR.fields_by_name.keys()

    # Make sure keyword arguments are valid
    for key, val in kwargs.items():
        if key not in field_names:
            raise TypeError("'{}' is an invalid keyword "
                            "argument for this function".format(key))
        if val is not None:
            field = getattr(message, key)
            if isinstance(val, google.protobuf.message.Message):
                field.CopyFrom(val)
            else:
                field.CopyFrom(val.export_proto())
            field.SetInParent()

    # Make sure default optimizer is set
    # TODO: This is a hack that should be removed when possible. LBANN
    # requires the prototext file to provide a default optimizer. It
    # would be better if LBANN used no optimizer if one isn't
    # provided.
    if not message.HasField('optimizer'):
        from lbann import Optimizer
        message.optimizer.CopyFrom(NoOptimizer().export_proto())
        message.optimizer.SetInParent()

    # Write to file
    with open(filename, 'wb') as f:
        if binary:
            f.write(message.SerializeToString())
        else:
            f.write(
                google.protobuf.text_format.MessageToString(
                    message, use_index_order=True).encode())


def text2bin(infile: str, outfile: str):
    """
    Converts a .prototext file to a .protobin file.
    """
    # Read file
    with open(infile, 'rb') as f:
        message = google.protobuf.text_format.Parse(f.read(),
                                                    lbann_pb2.LbannPB())

    # Write file
    with open(outfile, 'wb') as f:
        f.write(message.SerializeToString())


def bin2text(infile: str, outfile: str):
    """
    Converts a .prototext file to a .protobin file.
    """
    message = lbann_pb2.LbannPB()

    # Read file
    with open(infile, 'rb') as f:
        message.ParseFromString(f.read())

    # Write file
    with open(outfile, 'wb') as f:
        f.write(
            google.protobuf.text_format.MessageToString(
                message, use_index_order=True).encode())
