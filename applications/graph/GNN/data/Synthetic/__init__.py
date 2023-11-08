import lbann
import os.path as osp

data_dir = osp.dirname(osp.realpath(__file__))


def make_data_reader(classname,
                     sample='get_sample_func',
                     num_samples='num_samples_func',
                     sample_dims='sample_dims_func'):
    reader = lbann.reader_pb2.DataReader()
    _reader = reader.reader.add()
    _reader.name = 'python'
    _reader.role = 'train'
    _reader.shuffle = True
    _reader.fraction_of_data_to_use = 1.0
    _reader.python.module = classname
    _reader.python.module_dir = data_dir
    _reader.python.sample_function = sample
    _reader.python.num_samples_function = num_samples
    _reader.python.sample_dims_function = sample_dims
    return reader
