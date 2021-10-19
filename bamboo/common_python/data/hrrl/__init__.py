import gzip
import os
import os.path
import urllib.request

import google.protobuf.text_format

def make_data_reader(lbann):
    """Make Protobuf message for ATOM mpro data reader.


    """

    import lbann.contrib.lc.paths

    # Load data readers from prototext
    current_dir = os.path.dirname(os.path.realpath(__file__))
    # Load Protobuf message from file
    protobuf_file = os.path.join(current_dir,
                                 'probies_v2.prototext')

    message = lbann.lbann_pb2.LbannPB()
    with open(protobuf_file, 'r') as f:
        google.protobuf.text_format.Merge(f.read(), message)
    message = message.data_reader

    # Set paths
    return message
