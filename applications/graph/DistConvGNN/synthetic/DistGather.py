import argparse
import configparser
import lbann
import os
import lbann.contrib.launcher
import lbann.contrib.args


desc = ("Benchmarking code for distributed scatter with NVSHMEM")

parser = argparse.ArgumentParser(description=desc)

parser.add_argument('--job-name', action='store', default='distributed_scatter', type=str,
                    help='job name', metavar='NAME')

parser.add_argument('--mini-batch-size', action='store', default=4, type=int,
                    help='mini-batch size (default: 4)', metavar='NUM')

parser.add_argument('--num-epochs', action='store', default=2, type=int,
                    help='number of epochs (default: 2)', metavar='NUM')

parser.add_argument('--num-rows', action='store', default=32, type=int,
                    help='number of rows of the input matrix (default: 32)', metavar='NUM')

parser.add_argument('--num-cols', action='store', default=2, type=int,
                    help='number of columns of the input matrix (default: 2)', metavar='NUM')

parser.add_argument('--out-rows', action='store', default=16, type=int,
                    help='number of rows of the output matrix (default: 16)', metavar='NUM')

lbann.contrib.args.add_scheduler_arguments(parser)
args = parser.parse_args()

MINI_BATCH_SIZE = args.mini_batch_size
NUM_ROWS = args.num_rows
NUM_COLS = args.num_cols
OUT_ROWS = args.out_rows
NUM_NODES = args.nodes

#  Write to a config file so the synthetic data generator can generate appropriate data
config = configparser.ConfigParser()
config['DEFAULT'] = {"NumRows": f"{NUM_ROWS}",
                     "NumCols": f"{NUM_COLS}",
                     "OutRow": f"{OUT_ROWS}",
                     "Mode": "Gather"}

with open('data_config.ini', 'w') as configfile:
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
  _reader.percent_of_data_to_use = 1.0
  _reader.python.module = 'scatter_gather_dataset'
  _reader.python.module_dir = os.path.dirname(os.path.realpath(__file__))
  _reader.python.sample_function = 'get_sample'
  _reader.python.num_samples_function = 'num_train_samples'
  _reader.python.sample_dims_function = 'sample_dims'

  return reader


def main():
  matrix_size = NUM_ROWS * NUM_COLS
  indices_vector_size = OUT_ROWS

  x_weights = lbann.Weights(optimizer=lbann.SGD(),
                            initializer=lbann.ConstantInitializer(value=0.0),
                            name='input_weights')

  _inputs = lbann.Input(data_field='samples')

  sliced_inputs = lbann.Slice(_inputs, slice_points=(0, matrix_size, matrix_size + indices_vector_size))

  values = lbann.Identity(sliced_inputs)

  indices = lbann.Reshape(lbann.Identity(sliced_inputs), dims=[indices_vector_size, 1, 1], name='indices_array')
  
  # This is a hack to get around copying both parent tensors to DiHydrogen tensors in the Scatter layer
  # Currently we only support copying of only one of the parent tensors per distconv enabled layer

  indices = lbann.Relu(indices, parallel_strategy=create_parallel_strategy(4))

  values = lbann.Reshape(values, dims=[NUM_ROWS, NUM_COLS, 1], name='values_matrix')

  values_x = lbann.Sum(values, lbann.WeightsLayer(weights=x_weights, dims=[NUM_ROWS, NUM_COLS, 1]))

  # Only supporting fully distconv mode currently. So number of nodes == number of splits

  output = lbann.Gather(values_x,
                        indices,
                        axis=0,
                        name='output_matrix',
                        parallel_strategy=create_parallel_strategy(NUM_NODES))

  y = lbann.L2Norm2(output)

  print_model = lbann.CallbackPrintModelDescription()  # Prints initial Model after Setup

  gpu_usage = lbann.CallbackGPUMemoryUsage()

  dump_outputs = lbann.CallbackDumpOutputs(layers="output_matrix",
                                           batch_interval=5,
                                           directory=os.path.dirname(os.path.realpath(__file__)), format="csv")

  dump_inputs = lbann.CallbackDumpOutputs(layers="values_matrix",
                                          batch_interval=5,
                                          directory=os.path.dirname(os.path.realpath(__file__)), format="csv")

  callbacks = [print_model, dump_outputs, gpu_usage, dump_inputs]

  model = lbann.Model(2,
                      layers=lbann.traverse_layer_graph(_inputs),
                      objective_function=[y],
                      callbacks=callbacks)

  opt = lbann.NoOptimizer()
  data_reader = make_data_reader()
  trainer = lbann.Trainer(mini_batch_size=MINI_BATCH_SIZE)

  kwargs = lbann.contrib.args.get_scheduler_kwargs(args)
  lbann.contrib.launcher.run(trainer, model, data_reader, opt, **kwargs)


if __name__ == '__main__':
  main()
