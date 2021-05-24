import lbann
import lbann.contrib.launcher
import lbann.contrib.args

import argparse
import os
import configparser
import math

import data.LSC_PPQM4M
from lbann.util import str_list
from lbann.modules.graph import NNConv
from lbann.modules import ChannelwiseFullyConnectedModule

import numpy as np

desc = ("Training Edge-conditioned Graph Convolutional Model Using LBANN ")

parser = argparse.ArgumentParser(description=desc)

lbann.contrib.args.add_scheduler_arguments(parser)
lbann.contrib.args.add_optimizer_arguments(parser)

parser.add_argument(
    '--num-epochs', action='store', default=3, type=int,
    help='number of epochs (deafult: 3)', metavar='NUM')

parser.add_argument(
    '--mini-batch-size', action='store', default=2048, type=int,
    help="mini-batch size (default: 2048)", metavar='NUM')

parser.add_argument(
    '--num-edges', action='store', default=118, type=int,
    help='number of edges (deafult: 118)', metavar='NUM')

parser.add_argument(
    '--num-nodes', action='store', default=51, type=int,
    help='number of nodes (deafult: 51)', metavar='NUM')

parser.add_argument(
    '--num-node-features', action='store', default=9, type=int,
    help='number of node features (deafult: 9)', metavar='NUM')

parser.add_argument(
    '--num-edge-features', action='store', default=3, type=int,
    help='number of edge features (deafult: 3)', metavar='NUM')

parser.add_argument(
    '--num-out-features', action='store', default=32, type=int,
    help='number of node features for NNConv (deafult: 32)', metavar='NUM')

parser.add_argument(
    '--num-samples', action='store', default=3045360, type=int,
    help='number of Samples (deafult: 3045360)', metavar='NUM')


parser.add_argument(
    '--node-embeddings', action='store', default=100, type=int,
    help='dimensionality of node feature embedding (deafult: 100)', metavar='NUM')


parser.add_argument(
    '--edge-embeddings', action='store', default=16, type=int,
    help='dimensionality of edge feature embedding (deafult: 16)', metavar='NUM')

parser.add_argument(
    '--job-name', action='store', default="NN_Conv", type=str,
    help="Job name for scheduler", metavar='NAME')

args = parser.parse_args()

kwargs = lbann.contrib.args.get_scheduler_kwargs(args)

MINI_BATCH_SIZE = args.mini_batch_size
NUM_EPOCHS = args.num_epochs
JOB_NAME = args.job_name
NUM_NODES = 51
NUM_EDGES = 118
NUM_NODES_FEATURES = 9
NUM_EDGE_FEATURES = 3
NUM_OUT_FEATURES = args.num_out_features
NUM_SAMPLES = args.num_samples
EMBEDDING_DIM = args.node_embeddings
EDGE_EMBEDDING_DIM = args.edge_embeddings

# ----------------------------------------

# Generating configuration for dataset

# ----------------------------------------

config = configparser.ConfigParser()
config['Graph'] = {}
config['Graph']['num_nodes'] = str(NUM_NODES)
config['Graph']['num_edges'] = str(NUM_EDGES)
config['Graph']['num_node_features'] = str(NUM_NODES_FEATURES)
config['Graph']['num_edge_features'] = str(NUM_EDGE_FEATURES)
config['Graph']['num_samples'] = str(NUM_SAMPLES)

current_file = os.path.realpath(__file__)
app_dir = os.path.dirname(current_file)
_file_name = os.path.join(app_dir, 'config.ini')

with open(_file_name, 'w') as configfile:
    config.write(configfile)

os.environ['LBANN_LSC_CONFIG_FILE'] = _file_name


def _xavier_uniform_init(fan_in, fan_out):
    a = math.sqrt(6 / (fan_in + fan_out))
    return lbann.UniformInitializer(min=-a, max=a)


def BondEncoder(edge_feature_columns):
    # Courtesy of OGB
    bond_feature_dims = [5, 6, 2]
    _fan_in = bond_feature_dims[0]
    _fan_out = EDGE_EMBEDDING_DIM
    _embedding_weights = lbann.Weights(initializer=_xavier_uniform_init(_fan_in, _fan_out),
                                       name="bond_encoder_weights_{}".format(0))

    temp = lbann.Embedding(edge_feature_columns[0],
                           num_embeddings=bond_feature_dims[0],
                           embedding_dim=EDGE_EMBEDDING_DIM,
                           weights=_embedding_weights,
                           name="Bond_Embedding_0")

    for i in range(1, 3):
        _fan_in = bond_feature_dims[i]
        _fan_out = EDGE_EMBEDDING_DIM
        _embedding_weights = lbann.Weights(initializer=_xavier_uniform_init(_fan_in, _fan_out),
                                           name="bond_encoder_weights_{}".format(i))
        _temp2 = lbann.Embedding(edge_feature_columns[i],
                                 num_embeddings=bond_feature_dims[i],
                                 embedding_dim=EDGE_EMBEDDING_DIM,
                                 weights=_embedding_weights,
                                 name="Bond_Embedding_{}".format(i))
        temp = lbann.Sum(temp, _temp2)
    return temp


def AtomEncoder(node_feature_columns):
    # Courtesy of OGB
    atom_feature_dims = [119, 4, 12, 12, 10, 6, 6, 2, 2]

    _fan_in = atom_feature_dims[0]
    _fan_out = EDGE_EMBEDDING_DIM

    _embedding_weights = lbann.Weights(initializer=_xavier_uniform_init(_fan_in, _fan_out),
                                       name="atom_encoder_weights_{}".format(0))

    temp = lbann.Embedding(node_feature_columns[0],
                           num_embeddings=atom_feature_dims[0],
                           embedding_dim=EMBEDDING_DIM,
                           weights=_embedding_weights,
                           name="Atom_Embedding_0")
    for i in range(1, 9):
        _fan_in = atom_feature_dims[i]
        _fan_out = EDGE_EMBEDDING_DIM
        _embedding_weights = lbann.Weights(initializer=_xavier_uniform_init(_fan_in, _fan_out),
                                           name="atom_encoder_weights_{}".format(i))
        _temp2 = lbann.Embedding(node_feature_columns[i],
                                 num_embeddings=atom_feature_dims[i],
                                 embedding_dim=EMBEDDING_DIM,
                                 weights=_embedding_weights,
                                 name="Atom_Embedding_{}".format(i))
        temp = lbann.Sum(temp, _temp2)
    return temp


def _index_2d(index, dim1, dim2):
    """Modified index layer such that it works with value layer with dimension (dim1, dim2)
     for use with gather/scatter layers"""
    offset = lbann.Constant(value=dim2,
                            num_neurons=str_list([dim1, 1]))
    offset_target_indices = lbann.Multiply(index, offset)

    offset_values = np.tile(range(dim2), dim1)

    offset_mat_vals = lbann.Weights(
        initializer=lbann.ValueInitializer(values=str_list(offset_values)),
        optimizer=lbann.NoOptimizer())
    offset_mat = lbann.WeightsLayer(weights=offset_mat_vals, dims=str_list([dim1, dim2]))

    indices = lbann.Tessellate(offset_target_indices, dims=str_list([dim1, dim2]))

    modified_edge_indices = lbann.Reshape(lbann.Sum(offset_mat, indices), dims=str_list([dim1 * dim2]))

    return modified_edge_indices


def graph_data_splitter(_input):

    split_indices = []

    start_index = 0
    split_indices.append(start_index)

    node_feature = [NUM_NODES for i in range(1, NUM_NODES_FEATURES + 1)]

    split_indices.extend(node_feature)

    edge_features = [NUM_EDGES for i in range(1, NUM_EDGE_FEATURES + 1)]

    split_indices.extend(edge_features)

    edge_indices_sources = NUM_EDGES
    split_indices.append(edge_indices_sources)

    edge_indices_targets = NUM_EDGES
    split_indices.append(edge_indices_targets)

    node_mask = NUM_NODES
    split_indices.append(node_mask)

    target = 1
    split_indices.append(target)

    for i in range(1, len(split_indices)):
        split_indices[i] = split_indices[i] + split_indices[i - 1]

    graph_input = lbann.Slice(_input, axis=0,
                              slice_points=str_list(split_indices))

    node_feature_columns = [lbann.Reshape(lbann.Identity(graph_input),
                                          dims=str_list([NUM_NODES]),
                                          name="node_ft_{}_col".format(x)) for x in range(9)]

    edge_feature_columns = [lbann.Reshape(lbann.Identity(graph_input),
                                          dims=str_list([NUM_EDGES]),
                                          name="edge_ft_{}_col".format(x)) for x in range(3)]

    source_nodes = lbann.Reshape(lbann.Identity(graph_input),
                                 dims=str_list([NUM_EDGES, 1]),
                                 name="source_nodes")
    target_nodes = lbann.Reshape(lbann.Identity(graph_input),
                                 dims=str_list([NUM_EDGES, 1]),
                                 name="target_nodes")
    nodes_mask = lbann.Reshape(lbann.Identity(graph_input),
                               dims=str_list([NUM_NODES, 1]),
                               name="masked_nodes")
    label = lbann.Reshape(lbann.Identity(graph_input),
                          dims=str_list([1]),
                          name="Graph_Label")

    offset_values = np.tile(range(EMBEDDING_DIM), NUM_EDGES)

    offset_mat_vals = lbann.Weights(
        initializer=lbann.ValueInitializer(values=str_list(offset_values)),
        optimizer=lbann.NoOptimizer())

    offset_mat = lbann.WeightsLayer(weights=offset_mat_vals, dims=str_list([NUM_EDGES, EMBEDDING_DIM]), name="INDEX_OFFSET_MAP")

    indices = lbann.Tessellate(target_nodes, dims=str_list([NUM_EDGES, EMBEDDING_DIM]))

    modified_edge_indices = lbann.Reshape(lbann.Sum(offset_mat, indices), dims=str_list([NUM_EDGES * EMBEDDING_DIM]))

    neighbor_feature_dims = str_list([NUM_EDGES, 1, EMBEDDING_DIM])

    embedded_node_features = AtomEncoder(node_feature_columns)

    embedded_edge_features = BondEncoder(edge_feature_columns)

    neighbor_features = lbann.Reshape(embedded_node_features,
                                      dims=str_list([NUM_NODES * EMBEDDING_DIM]),
                                      name='Flattened_Neighbor_features')

    neighbor_features = lbann.Gather(neighbor_features, modified_edge_indices)

    neighbor_feature_mat = lbann.Reshape(neighbor_features,
                                         dims=neighbor_feature_dims)

    return \
        embedded_node_features, neighbor_feature_mat, embedded_edge_features, source_nodes, label


def reduction(graph_feature_matrix, channels):
    vector_size = str_list([1, NUM_NODES])
    reduction_vector = lbann.Constant(value=1,
                                      num_neurons=vector_size,
                                      name='Sum_Vector')
    reduced_features = lbann.MatMul(reduction_vector, graph_feature_matrix,
                                    name='Node_Feature_Reduction')
    reduced_features = lbann.Reshape(reduced_features,
                                     dims=str_list([channels]))
    return reduced_features


def NNConvLayer(node_features,
                neighbor_features,
                edge_features,
                edge_index,
                in_channel,
                out_channel,
                layer_name=0):

    FC = ChannelwiseFullyConnectedModule

    k_1 = math.sqrt(1 / in_channel)
    k_2 = math.sqrt(1 / 64)
    k_3 = math.sqrt(1 / 32)
    nn_sq_1_weight = lbann.Weights(initializer=lbann.UniformInitializer(min=-k_1, max=k_1),
                                   name="gnn_weights_{}".format(0))
    nn_sq_2_weight = lbann.Weights(initializer=lbann.UniformInitializer(min=-k_2, max=k_2),
                                   name="gnn_weights_weights_{}".format(1))
    nn_sq_3_weight = lbann.Weights(initializer=lbann.UniformInitializer(min=-k_3, max=k_3),
                                   name="gnn_weights_weights_{}".format(2))

    sequential_nn = \
        [FC(64, weights=[nn_sq_1_weight], name="NN_SQ_1", bias=True),
         lbann.Relu,
         FC(32, weights=[nn_sq_2_weight], name="NN_SQ_2", bias=True),
         lbann.Relu,
         FC(out_channel * in_channel, weights=[nn_sq_3_weight], name="NN_SQ_3", bias=True)]

    nn_conv = NNConv(sequential_nn,
                     NUM_NODES,
                     NUM_EDGES,
                     in_channel,
                     out_channel)

    out = nn_conv(node_features,
                  neighbor_features,
                  edge_features,
                  edge_index)
    return out


def make_model():
    in_channel = EMBEDDING_DIM
    out_channel = NUM_OUT_FEATURES
    output_dimension = 1

    _input = lbann.Input(target_mode='N/A')
    node_feature_mat, neighbor_feature_mat, edge_feature_mat, edge_indices, target = \
        graph_data_splitter(_input)

    modified_edge_indices = _index_2d(edge_indices, NUM_EDGES, out_channel)

    x = NNConvLayer(node_feature_mat,
                    neighbor_feature_mat,
                    edge_feature_mat,
                    modified_edge_indices,
                    in_channel,
                    out_channel)

    for i, num_neurons in enumerate([256, 128, 32, 8], 1):
        x = lbann.FullyConnected(x,
                                 num_neurons=num_neurons,
                                 name="hidden_layer_{}".format(i))
        x = lbann.Relu(x, name='hidden_layer_{}_activation'.format(i))
    x = lbann.FullyConnected(x,
                             num_neurons=output_dimension,
                             name="output")

    loss = lbann.MeanAbsoluteError(x, target)

    layers = lbann.traverse_layer_graph(_input)
    training_output = lbann.CallbackPrint(interval=1,
                                          print_global_stat_only=False)
    gpu_usage = lbann.CallbackGPUMemoryUsage()
    timer = lbann.CallbackTimer()

    callbacks = [training_output, gpu_usage, timer]
    model = lbann.Model(NUM_EPOCHS,
                        layers=layers,
                        objective_function=loss,
                        callbacks=callbacks)
    return model


model = make_model()
optimizer = lbann.SGD(learn_rate=1e-4)
data_reader = data.LSC_PPQM4M.make_data_reader("LSC_FULL_DATA")
trainer = lbann.Trainer(mini_batch_size=MINI_BATCH_SIZE)


lbann.contrib.launcher.run(trainer,
                           model,
                           data_reader,
                           optimizer,
                           job_name=JOB_NAME,
                           **kwargs)
