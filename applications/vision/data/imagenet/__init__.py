from typing import Any

import os
import os.path

import google.protobuf.text_format
import lbann
import lbann.contrib.launcher


def make_data_reader_imagenet(num_classes: int = 1000) -> Any:
    """Load and set up the ImageNet data reader."""
    current_dir = os.path.dirname(os.path.realpath(__file__))
    protobuf_file = os.path.join(current_dir, 'data_reader.prototext')
    message = lbann.lbann_pb2.LbannPB()
    with open(protobuf_file, 'r') as f:
        google.protobuf.text_format.Merge(f.read(), message)
    message = message.data_reader

    # Paths to ImageNet data
    # Note: Paths are only known for some compute centers
    compute_center = lbann.contrib.launcher.compute_center()
    if compute_center == 'lc':
        from lbann.contrib.lc.paths import imagenet_dir, imagenet_labels
        train_data_dir = imagenet_dir(data_set='train',
                                      num_classes=num_classes)
        train_label_file = imagenet_labels(data_set='train',
                                           num_classes=num_classes)
        test_data_dir = imagenet_dir(data_set='val',
                                     num_classes=num_classes)
        test_label_file = imagenet_labels(data_set='val',
                                          num_classes=num_classes)
    elif compute_center == 'nersc':
        from lbann.contrib.nersc.paths import imagenet_dir, imagenet_labels
        train_data_dir = imagenet_dir(data_set='train')
        train_label_file = imagenet_labels(data_set='train')
        test_data_dir = imagenet_dir(data_set='val')
        test_label_file = imagenet_labels(data_set='val')
    else:
        raise RuntimeError(f'ImageNet data paths are unknown for current compute center ({compute_center})')

    # Check that data paths are accessible
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


def make_data_reader_synthetic(num_classes: int = 1000) -> Any:
    """Load and set up a synthetic data reader matching ImageNet."""
    current_dir = os.path.dirname(os.path.realpath(__file__))
    protobuf_file = os.path.join(current_dir, 'data_reader_synthetic.prototext')
    message = lbann.lbann_pb2.LbannPB()
    with open(protobuf_file, 'r') as f:
        google.protobuf.text_format.Merge(f.read(), message)
    message = message.data_reader

    return message

def make_data_reader(
        num_classes: int = 1000,
        synthetic: bool = False) -> Any:
    if synthetic:
        return make_data_reader_synthetic(num_classes=num_classes)
    else:
        return make_data_reader_imagenet(num_classes=num_classes)
