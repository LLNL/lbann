import os
import os.path

import google.protobuf.text_format
import lbann
import lbann.contrib.lc.paths

def make_data_reader(num_classes=1000):

    # Load Protobuf message from file
    current_dir = os.path.dirname(os.path.realpath(__file__))
    protobuf_file = os.path.join(current_dir, 'data_reader.prototext')
    message = lbann.lbann_pb2.LbannPB()
    with open(protobuf_file, 'r') as f:
        google.protobuf.text_format.Merge(f.read(), message)
    message = message.data_reader

    # Check if data paths are accessible
    train_data_dir = lbann.contrib.lc.paths.imagenet_dir(data_set='train',
                                                         num_classes=num_classes)
    train_label_file = lbann.contrib.lc.paths.imagenet_labels(data_set='train',
                                                              num_classes=num_classes)
    test_data_dir = lbann.contrib.lc.paths.imagenet_dir(data_set='val',
                                                        num_classes=num_classes)
    test_label_file = lbann.contrib.lc.paths.imagenet_labels(data_set='val',
                                                             num_classes=num_classes)
    if not os.path.isdir(train_data_dir):
        raise FileNotFoundError('could not access {}'.format(train_data_dir))
    if not os.path.isfile(train_label_file):
        raise FileNotFoundError('could not access {}'.format(train_label_file))
    if not os.path.isdir(test_data_dir):
        raise FileNotFoundError('could not access {}'.format(test_data_dir))
    if not os.path.isfile(test_label_file):
        raise FileNotFoundError('could not access {}'.format(test_label_file))

    # Set paths
    message.reader[0].data_filedir = train_data_dir
    message.reader[0].data_filename = train_label_file
    message.reader[1].data_filedir = test_data_dir
    message.reader[1].data_filename = test_label_file

    return message
