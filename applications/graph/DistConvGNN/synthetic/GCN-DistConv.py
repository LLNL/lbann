import argparse
import configparser
import lbann
import os
import lbann.contrib.launcher
import lbann.contrib.args


desc = ("Benchmarking code for distributed gather with NVSHMEM")

parser = argparse.ArgumentParser(description=desc)

parser.add_argument('--job-name', action='store', default='distributed_gather', type=str,
                    help='job name', metavar='NAME')

parser.add_argument('--mini-batch-size', action='store', default=4, type=int,
                    help='mini-batch size (default: 4)', metavar='NUM')

parser.add_argument('--num-epochs', action='store', default=2, type=int,
                    help='number of epochs (default: 2)', metavar='NUM')

parser.add_argument('--num-vertices', action='store', default=32, type=int,
                    help='number of vertice per graph (default: 32)', metavar='NUM')

parser.add_argument('--num-features', action='store', default=16, type=int,
                    help='size of node features vector (default: 16)', metavar='NUM')

parser.add_argument('--num-edges', action='store', default=128, type=int,
                    help='number of edges per graph (default: 128)', metavar='NUM')

parser.add_argument('--enable-distconv', action='store_true')
parser.add_argument('--disable-distconv', dest='enable-distconv', action='store_false')
parser.set_defaults(feature=True)

lbann.contrib.args.add_scheduler_arguments(parser)
args = parser.parse_args()

MINI_BATCH_SIZE = args.mini_batch_size
NUM_VERTICES = args.num_vertices
NUM_FEATS = args.num_features
NUM_EDGES = args.num_edges
NUM_NODES = args.nodes
DISTCONV_ENABLED = args.enable_distconv
NUM_EPOCHS = args.num_epochs

#  Write to a config file so the synthetic data generator can generate appropriate data
config = configparser.ConfigParser()
config['DEFAULT'] = {"NumVertices": f"{NUM_VERTICES}",
                     "NumFeats": f"{NUM_FEATS}",
                     "NumEdges": f"{NUM_EDGES}"}

with open('gcn_data_config.ini', 'w') as configfile:
  config.write(configfile)


def create_parallel_strategy(num_channel_groups):
    return {"channel_groups": num_channel_groups,
            "filter_groups": num_channel_groups}


def make_data_reader():
  reader = lbann.reader_pb2.DataReader()
  _reader = reader.reader.add()
  _reader.name = 'python'
  _reader.role = 'train'
  _reader.shuffle = False
  _reader.fraction_of_data_to_use = 1.0
  _reader.python.module = 'gcn_dataset'
  _reader.python.module_dir = os.path.dirname(os.path.realpath(__file__))
  _reader.python.sample_function = 'get_sample'
  _reader.python.num_samples_function = 'num_train_samples'
  _reader.python.sample_dims_function = 'sample_dims'

  return reader


def main():
  node_feature_mat_size = NUM_VERTICES * NUM_FEATS
  edge_index_mat_size = NUM_EDGES * 2
  targets = 10

  slice_points = [0,
                  node_feature_mat_size,
                  node_feature_mat_size + edge_index_mat_size,
                  node_feature_mat_size + edge_index_mat_size + targets]

  if DISTCONV_ENABLED:
    node_ft_dims = [NUM_VERTICES, NUM_FEATS, 1]
    source_ind_dims = [NUM_EDGES, 1, 1]
    target_ind_dims = [NUM_EDGES, 1, 1]

  else:
    node_ft_dims = [NUM_VERTICES, NUM_FEATS]
    source_ind_dims = [NUM_EDGES, 1]
    target_ind_dims = [NUM_EDGES, 1]

  _inputs = lbann.Input(data_field='samples')
  sliced_inputs = lbann.Slice(_inputs, slice_points=slice_points)

  node_features = lbann.Reshape(lbann.Identity(sliced_inputs), dims=node_ft_dims, name='node_feature_matrix')

  edge_indices = lbann.Reshape(lbann.Identity(sliced_inputs), dims=[2, NUM_EDGES])

  targets = lbann.Identity(sliced_inputs)

  sliced_indices = lbann.Slice(edge_indices, axis=0, slice_points=[NUM_EDGES])

  source_indices = lbann.Reshape(lbann.Identity(sliced_indices), dims=source_ind_dims)
  target_indices = lbann.Reshape(lbann.Identity(sliced_indices), dims=target_ind_dims)

  if DISTCONV_ENABLED:
    x = lbann.ChannelwiseFullyConnected(node_features, parallel_strategy=create_parallel_strategy(NUM_NODES))
    x = lbann.Relu(x, parallel_strategy=create_parallel_strategy(NUM_NODES))
    x = lbann.Gather(x, source_indices, parallel_strategy=create_parallel_strategy(NUM_NODES))
    x = lbann.Scatter(x, target_indices, dims=node_ft_dims, parallel_strategy=create_parallel_strategy(NUM_NODES))
  else:
    x = lbann.ChannelwiseFullyConnected(node_features)
    x = lbann.Relu()
    x = lbann.Gather(x, source_indices)
    x = lbann.Scatter(x, target_indices, dims=node_ft_dims)

  average_vector = lbann.Constant(value=1 / NUM_VERTICES, num_neurons=[1, NUM_VERTICES], name="Average_Vector")
  x = lbann.MatMul(average_vector, x, name="Node_Feature_Reduction")
  x = lbann.FullyConnected(x, num_neurons=targets)

  probs = lbann.Softmax(x, name="Softmax")
  loss = lbann.CrossEntropy(probs, targets, name="Cross_Entropy_Loss")

  print_model = lbann.CallbackPrintModelDescription()  # Prints initial Model after Setup
  gpu_usage = lbann.CallbackGPUMemoryUsage()

  callbacks = [print_model, gpu_usage]
  model = lbann.Model(NUM_EPOCHS,
                      layers=lbann.traverse_layer_graph(_inputs),
                      objective_function=loss,
                      callbacks=callbacks)

  opt = lbann.SGD(learn_rate=1e-3)
  data_reader = make_data_reader()
  trainer = lbann.Trainer(mini_batch_size=MINI_BATCH_SIZE)

  kwargs = lbann.contrib.args.get_scheduler_kwargs(args)
  lbann.contrib.launcher.run(trainer, model, data_reader, opt, **kwargs)


if __name__ == '__main__':
  main()
