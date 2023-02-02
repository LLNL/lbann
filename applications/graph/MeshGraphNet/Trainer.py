import lbann
import lbann.contrib.launcher
import lbann.contrib.args
import argparse

import os.path as osp
from .GNN import LBANN_GNN_Model

data_dir = osp.dirname(osp.realpath(__file__))


desc = ("Training a Mesh Graph Neural Network Model Using LBANN")

parser = argparse.ArgumentParser(description=desc)

lbann.contrib.args.add_scheduler_arguments(parser)
lbann.contrib.args.add_optimizer_arguments(parser)

parser.add_argument(
    '--num-epochs', action='store', default=3, type=int,
    help='number of epochs (deafult: 3)', metavar='NUM')

parser.add_argument(
    '--mini-batch-size', action='store', default=256, type=int,
    help="mini-batch size (default: 256)", metavar='NUM')

parser.add_argument(
    '--job-name', action='store', default="MGN", type=str,
    help="Job name for scheduler", metavar='NAME')

args = parser.parse_args()
kwargs = lbann.contrib.args.get_scheduler_kwargs(args)

# Some training parameters

MINI_BATCH_SIZE = args.mini_batch_size
NUM_EPOCHS = args.num_epochs
JOB_NAME = args.job_name

# Some synthetic attributes to get the model running

NUM_NODES = 100
NUM_EDGES = 1000
NODE_FEATS = 5
EDGE_FEATS = 3
OUT_FEATS = 3

def make_data_reader(classname,
                     sample='get_sample_func',
                     num_samples='num_samples_func',
                     sample_dims='sample_dims_func',
                     validation_percent=0.1):
    reader = lbann.reader_pb2.DataReader()
    _reader = reader.reader.add()
    _reader.name = 'python'
    _reader.role = 'train'
    _reader.shuffle = True
    _reader.percent_of_data_to_use = 1.0
    _reader.validation_percent = validation_percent
    _reader.python.module = classname
    _reader.python.module_dir = data_dir
    _reader.python.sample_function = sample
    _reader.python.num_samples_function = num_samples
    _reader.python.sample_dims_function = sample_dims
    return reader

def main():
  # Use the defaults for the other parameters
  model = LBANN_GNN_Model(num_nodes=NUM_NODES,
                          num_edges=NUM_EDGES,
                          in_dim_node=NODE_FEATS,
                          in_dim_edge=EDGE_FEATS,
                          out_dim=OUT_FEATS,
                          num_epochs=NUM_EPOCHS)

  optimizer = lbann.SGD(learn_rate=1e-4)
  data_reader = make_data_reader("SyntheticData")
  trainer = lbann.Trainer(mini_batch_size=MINI_BATCH_SIZE)

  lbann.contrib.launcher.run(trainer,
                            model,
                            data_reader,
                            optimizer,
                            job_name=JOB_NAME,
                            **kwargs)

if __name__ == '__main__':
  main()