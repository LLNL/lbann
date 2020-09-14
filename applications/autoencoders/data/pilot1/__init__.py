import os
import os.path

import google.protobuf.text_format
import lbann

# Paths
data_dir = os.path.dirname(os.path.realpath(__file__))

# FIXME: Should I add anything to check paths like in cifar10, or
# download anything like in mnist?

def make_data_reader(data_reader_file='data_reader_candle_pilot1.prototext'):

    # Load Protobuf message from file
    protobuf_file = os.path.join(data_dir, data_reader_file)
    message = lbann.lbann_pb2.LbannPB()
    with open(protobuf_file, 'r') as f:
        google.protobuf.text_format.Merge(f.read(), message)
    message = message.data_reader

    # Set paths
    for reader in message.reader:
        reader.data_filedir = data_dir

    return message
