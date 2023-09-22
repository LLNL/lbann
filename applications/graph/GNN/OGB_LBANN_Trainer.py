import lbann
import lbann.contrib.launcher
import lbann.contrib.args

import argparse
import os
import configparser
import math
import data.LSC_PPQM4M
from lbann.modules.graph import NNConv
from lbann.modules import ChannelwiseFullyConnectedModule

import numpy as np

from NNConvModel import make_model
desc = ("Training Edge-conditioned Graph Convolutional Model Using LBANN ")

parser = argparse.ArgumentParser(description=desc)

lbann.contrib.args.add_scheduler_arguments(parser, 'NN_Conv')
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
    '--num-samples', action='store', default=100000, type=int,
    help='number of Samples (deafult: 100000)', metavar='NUM')

parser.add_argument(
    '--node-embeddings', action='store', default=100, type=int,
    help='dimensionality of node feature embedding (deafult: 100)', metavar='NUM')

parser.add_argument(
    '--edge-embeddings', action='store', default=16, type=int,
    help='dimensionality of edge feature embedding (deafult: 16)', metavar='NUM')

parser.add_argument(
    '--enable-distconv', action='store_true',
    help="Enables distconv-mode for graph kernels")

parser.add_argument(
    '--process-groups', action='store', default=0, type=int,
    help="Number of parallel groups for distconv", metavar='NUM')


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

# ----------------------------------------

# Enabling distconv

# ----------------------------------------

if (not args.enable_distconv and args.process_groups > 0):
    raise ValueError('Cannot have non-zero process-groups with distconv disabled. Enable distconv with --distconv')

NUM_PROCESS_GROUPS = args.process_groups

# ---------------------------------------------------

# Create model, data reader, optimizer, and trainer

# ---------------------------------------------------

model = make_model(NUM_NODES,
                   NUM_EDGES,
                   NUM_NODES_FEATURES,
                   NUM_EDGE_FEATURES,
                   EMBEDDING_DIM,
                   EDGE_EMBEDDING_DIM,
                   NUM_OUT_FEATURES,
                   NUM_EPOCHS,
                   NUM_PROCESS_GROUPS)

optimizer = lbann.SGD(learn_rate=1e-4)
data_reader = data.LSC_PPQM4M.make_data_reader("LSC_100K")
trainer = lbann.Trainer(mini_batch_size=MINI_BATCH_SIZE)


lbann.contrib.launcher.run(trainer,
                           model,
                           data_reader,
                           optimizer,
                           job_name=JOB_NAME,
                           **kwargs)
