import configparser
import os.path
import lbann

from util import str_list

def make_offline_data_reader():
    reader = lbann.reader_pb2.DataReader()
    _reader = reader.reader.add()
    _reader.name = 'python'
    _reader.role = 'train'
    _reader.shuffle = True
    _reader.percent_of_data_to_use = 1.0
    _reader.python.module = 'dataset'
    _reader.python.module_dir = os.path.dirname(os.path.realpath(__file__))
    _reader.python.sample_function = 'get_sample'
    _reader.python.num_samples_function = 'num_samples'
    _reader.python.sample_dims_function = 'sample_dims'
    return reader

def make_online_data_reader(config):

    # Get parameters
    embedding_weights = 'generator_embeddings'
    graph_file = config.get('Graph', 'file')
    num_vertices = config.getint('Graph', 'num_vertices')
    motif_file = config.get('Motifs', 'file')
    motif_size = config.getint('Motifs', 'motif_size')
    walk_length = config.getint('Walks', 'walk_length')
    walks_per_vertex = config.getint('Walks', 'num_walkers')
    start_vertices_file = config.get('Walks', 'start_vertices_file')
    path_limit = config.getint('Walks', 'path_limit')
    mini_batch_size = config.getint('Embeddings', 'mini_batch_size')
    sgd_steps_per_epoch = config.getint('Embeddings', 'sgd_steps_per_epoch')

    # Configure data reader
    reader = lbann.reader_pb2.DataReader()
    _reader = reader.reader.add()
    _reader.name = 'communitygan'
    _reader.role = 'train'
    _reader.percent_of_data_to_use = 1.0
    _reader.communitygan.embedding_weights = embedding_weights
    _reader.communitygan.motif_file = motif_file
    _reader.communitygan.graph_file = graph_file
    _reader.communitygan.start_vertices_file = start_vertices_file
    _reader.communitygan.num_vertices = num_vertices
    _reader.communitygan.motif_size = motif_size
    _reader.communitygan.walk_length = walk_length
    _reader.communitygan.walks_per_vertex = walks_per_vertex
    _reader.communitygan.path_limit = path_limit
    _reader.communitygan.epoch_size = sgd_steps_per_epoch * mini_batch_size
    return reader
