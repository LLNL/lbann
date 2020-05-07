"""Learn embedding weights with LBANN."""
import argparse
import os.path

import lbann
import lbann.contrib.launcher
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
    '--work-dir', action='store', default=None, type=str,
    help='working directory', metavar='DIR')
args = parser.parse_args()

# ----------------------------------
# Embedding weights
# ----------------------------------

encoder_embeddings_weights = lbann.Weights(
    initializer=lbann.NormalInitializer(
        mean=0, standard_deviation=1/args.latent_dim,
    ),
    name='embeddings',
)
decoder_embeddings_weights = lbann.Weights(
    initializer=lbann.ConstantInitializer(value=0),
    name='decoder_embeddings',
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
input_ = lbann.Identity(lbann.Input())
input_slice = lbann.Slice(
    input_,
    slice_points=f'0 {num_negative_samples+1} {input_size}'
)
decoder_embeddings = lbann.Embedding(
    input_slice,
    weights=decoder_embeddings_weights,
    num_embeddings=num_graph_nodes,
    embedding_dim=args.latent_dim,
)
encoder_embeddings = lbann.Embedding(
    input_slice,
    weights=encoder_embeddings_weights,
    num_embeddings=num_graph_nodes,
    embedding_dim=args.latent_dim,
)

# Skip-Gram with negative sampling
preds = lbann.MatMul(decoder_embeddings, encoder_embeddings, transpose_b=True)
preds_slice = lbann.Slice(
    preds,
    axis=0,
    slice_points=f'0 {num_negative_samples} {num_negative_samples+1}')
preds_negative = lbann.Identity(preds_slice)
preds_positive = lbann.Identity(preds_slice)
obj_positive = lbann.LogSigmoid(preds_positive)
obj_positive = lbann.Reduction(obj_positive, mode='sum')
obj_negative = lbann.WeightedSum(preds_negative, scaling_factors='-1')
obj_negative = lbann.LogSigmoid(obj_negative)
obj_negative = lbann.Reduction(obj_negative, mode='sum')
obj = [
    lbann.LayerTerm(obj_positive, scale=-1),
    lbann.LayerTerm(obj_negative, scale=-1/num_negative_samples),
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

# Create optimizer
# Note: Learning rate in original word2vec is 0.025
learning_rate = args.learning_rate
if learning_rate < 0:
    learning_rate = 0.025 * args.mini_batch_size
opt = lbann.SGD(learn_rate=learning_rate)

# Create LBANN objects
trainer = lbann.Trainer(mini_batch_size=args.mini_batch_size)
callbacks = [
    lbann.CallbackPrint(),
    lbann.CallbackTimer(),
    lbann.CallbackDumpWeights(basename='embeddings',
                              epoch_interval=args.num_epochs),
]
model = lbann.Model(args.num_epochs,
                    layers=lbann.traverse_layer_graph(input_),
                    objective_function=obj,
                    callbacks=callbacks)

# Run LBANN
kwargs = lbann.contrib.args.get_scheduler_kwargs(args)
lbann.contrib.launcher.run(trainer, model, reader, opt,
                           job_name=args.job_name,
                           work_dir=args.work_dir,
                           overwrite_script=True,
                           **kwargs)
