import os.path
import lbann

def make_online_data_reader(
        graph_file,
        epoch_size,
        walk_length=80,
        return_param=0.25,
        inout_param=0.25,
        num_negative_samples=5,
):

    reader = lbann.reader_pb2.DataReader()
    _reader = reader.reader.add()
    _reader.name = 'node2vec'
    _reader.role = 'train'
    _reader.shuffle = True
    _reader.fraction_of_data_to_use = 1.0
    _reader.node2vec.graph_file = graph_file
    _reader.node2vec.epoch_size = epoch_size
    _reader.node2vec.walk_length = walk_length
    _reader.node2vec.return_param = return_param
    _reader.node2vec.inout_param = inout_param
    _reader.node2vec.num_negative_samples = num_negative_samples

    return reader

def make_offline_data_reader():

    reader = lbann.reader_pb2.DataReader()
    _reader = reader.reader.add()
    _reader.name = 'python'
    _reader.role = 'train'
    _reader.shuffle = True
    _reader.fraction_of_data_to_use = 1.0
    _reader.python.module = 'offline_walks'
    _reader.python.module_dir = os.path.dirname(os.path.realpath(__file__))
    _reader.python.sample_function = 'get_sample'
    _reader.python.num_samples_function = 'num_samples'
    _reader.python.sample_dims_function = 'sample_dims'
    return reader
