import gzip
import os
import os.path
import urllib.request

import google.protobuf.text_format

def make_data_reader(lbann):
    """Make Protobuf message for MNIST data reader.

    MNIST data is downloaded if needed.

    """

    # TODO (tym): Figure out how to switch between LBANN builds. See
    # GitHub Issue #1289.
    import lbann.contrib.lc.paths

    # Load data readers from prototext
    current_dir = os.path.dirname(os.path.realpath(__file__))
    # Load Protobuf message from file
    protobuf_file = os.path.join(current_dir,
                                 'data_reader.prototext')

    message = lbann.lbann_pb2.LbannPB()
    with open(protobuf_file, 'r') as f:
        google.protobuf.text_format.Merge(f.read(), message)
    message = message.data_reader

    # Set paths
    for reader in message.reader:
        reader.data_filedir = lbann.contrib.lc.paths.mnist_dir()

    return message
