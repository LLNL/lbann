import os
import os.path

import google.protobuf.text_format
import lbann
import lbann.contrib.lc.paths

def make_data_reader(num_classes=10):

    # Load Protobuf message from file
    current_dir = os.path.dirname(os.path.realpath(__file__))
    protobuf_file = os.path.join(current_dir, 'data_reader.prototext')
    message = lbann.lbann_pb2.LbannPB()
    with open(protobuf_file, 'r') as f:
        google.protobuf.text_format.Merge(f.read(), message)
    message = message.data_reader

    # Check if data paths are accessible
    data_dir = lbann.contrib.lc.paths.cifar10_dir()

    if not os.path.isdir(data_dir):
        raise FileNotFoundError('could not access {}'.format(data_dir))

    # Set paths
    message.reader[0].data_filedir = data_dir
    message.reader[1].data_filedir = data_dir

    return message
