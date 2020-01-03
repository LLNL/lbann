"""Learn embedding weights with LBANN."""
import argparse
import os.path

import lbann
import lbann.contrib.lc.launcher
import lbann.contrib.args

import dataset
from utils import make_iterable, str_list
import utils.snap

# ----------------------------------
# Options
# ----------------------------------

# Command-line arguments
parser = argparse.ArgumentParser()
lbann.contrib.args.add_scheduler_arguments(parser)
parser.add_argument(
    '--job-name', action='store', default='lbann_node2vec', type=str,
    help='job name', metavar='NAME')
parser.add_argument(
    '--mini-batch-size', action='store', default=256, type=int,
    help='mini-batch size (default: 256)', metavar='NUM')
parser.add_argument(
    '--num-epochs', action='store', default=1, type=int,
    help='number of epochs (default: 1)', metavar='NUM')
parser.add_argument(
    '--latent-dim', action='store', default=128, type=int,
    help='latent space dimensions (default: 128)', metavar='NUM')
parser.add_argument(
    '--learning-rate', action='store', default=-1, type=float,
    help='learning rate (default: 0.025*mbsize)', metavar='VAL')
parser.add_argument(
    '--experiment-dir', action='store', default=None, type=str,
    help='directory for experiment artifacts', metavar='DIR')
args = parser.parse_args()

# ----------------------------------
# Embedding weights
# ----------------------------------

embeddings = lbann.Weights(
    initializer=lbann.NormalInitializer(mean=0, standard_deviation=1),
    name='embeddings',
)

# ----------------------------------
# Construct layer graph
# ----------------------------------

# Properties of graph and random walk
num_graph_nodes = dataset.max_graph_node_id() + 1
walk_length = dataset.walk_context_length
num_negative_samples = dataset.num_negative_samples
input_size = dataset.sample_dims()[0]

# Embedding vectors, including negative sampling
# Note: Input is sequence of graph node IDs
input_embeddings = lbann.Embedding(
    lbann.Input(),
    weights=embeddings,
    num_embeddings=num_graph_nodes,
    embedding_dim=args.latent_dim,
)

# Skip-Gram with negative sampling
input_embeddings_slice = lbann.Slice(
    input_embeddings,
    axis=0,
    slice_points=f'0 {num_negative_samples+1} {input_size}'
)
left_embeddings = lbann.Identity(input_embeddings_slice)
right_embeddings = lbann.Identity(input_embeddings_slice)
preds = lbann.MatMul(left_embeddings, right_embeddings, transpose_b=True)
preds = lbann.LogSigmoid(preds)
preds = lbann.Slice(preds,
                    axis=0,
                    slice_points=f'0 {num_negative_samples} {num_negative_samples+1}')
preds_negative = lbann.Reduction(preds, mode='average')
preds_positive = lbann.Reduction(preds, mode='average')
obj = [
    lbann.LayerTerm(preds_positive, scale=-1),
    lbann.LayerTerm(preds_negative, scale=1),
]

# ----------------------------------
# Create data reader
# ----------------------------------

reader = lbann.reader_pb2.DataReader()
_reader = reader.reader.add()
_reader.name = 'python'
_reader.role = 'train'
_reader.percent_of_data_to_use = 1.0
_reader.python.module = 'dataset'
_reader.python.module_dir = os.path.dirname(os.path.realpath(__file__))
_reader.python.sample_function = 'get_sample'
_reader.python.num_samples_function = 'num_samples'
_reader.python.sample_dims_function = 'sample_dims'

# ----------------------------------
# Run LBANN
# ----------------------------------

# Set learning rate
# Note: Learning rate in original word2vec is 0.025
learning_rate = args.learning_rate
if learning_rate < 0:
    learning_rate = 0.025 * args.mini_batch_size

# Create LBANN objects
trainer = lbann.Trainer()
callbacks = [
    lbann.CallbackPrint(),
    lbann.CallbackTimer(),
    lbann.CallbackDumpWeights(basename='embeddings',
                              epoch_interval=args.num_epochs),
    lbann.CallbackPrintModelDescription(),
]
model = lbann.Model(args.mini_batch_size,
                    args.num_epochs,
                    layers=lbann.traverse_layer_graph(input_embeddings),
                    objective_function=obj,
                    callbacks=callbacks)
opt = lbann.SGD(learn_rate=learning_rate)

# Run LBANN
kwargs = lbann.contrib.args.get_scheduler_kwargs(args)
lbann.contrib.lc.launcher.run(trainer, model, reader, opt,
                              job_name=args.job_name,
                              experiment_dir=args.experiment_dir,
                              overwrite_script=True,
                              **kwargs)
