import os
import os.path

import google.protobuf.text_format
import lbann

def make_data_reader(data_reader_file='data_reader_candle_pilot1.prototext'):

    # Load Protobuf message from file
    protobuf_file = os.path.join(data_dir, data_reader_file)
    message = lbann.lbann_pb2.LbannPB()
    with open(protobuf_file, 'r') as f:
        google.protobuf.text_format.Merge(f.read(), message)
    message = message.data_reader

    return message
