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
    '--experiment-dir', action='store', default=None, type=str,
    help='directory for experiment artifacts', metavar='DIR')
args = parser.parse_args()

# ----------------------------------
# Embedding weights
# ----------------------------------

embeddings = lbann.Weights(initializer=lbann.NormalInitializer(mean=0,
                                                               standard_deviation=1),
                           name='embeddings')

# ----------------------------------
# Construct layer graph
# ----------------------------------

# Properties of graph and random walk
num_graph_nodes = dataset.max_graph_node_id() + 1
walk_length = dataset.walk_context_length
num_negative_samples = dataset.num_negative_samples

# Input is a sequence of graph node IDs
input_size = dataset.sample_dims()[0]
input_ = lbann.Identity(lbann.Input())
input_slice = lbann.Slice(input_,
                          slice_points=str_list(range(input_size+1)))
walk = []
negative_samples = []
for _ in range(walk_length):
    walk.append(lbann.Identity(input_slice))
for _ in range(num_negative_samples):
    negative_samples.append(lbann.Identity(input_slice))

# Embedding vectors, including negative sampling
source_embedding = lbann.Embedding(walk[0],
                                   weights=embeddings,
                                   num_embeddings=num_graph_nodes,
                                   embedding_dim=args.latent_dim)
source_embedding = lbann.Reshape(source_embedding,
                                 dims=f'{args.latent_dim} 1')
positive_embeddings = [lbann.Embedding(node,
                                       weights=embeddings,
                                       num_embeddings=num_graph_nodes,
                                       embedding_dim=args.latent_dim)
                       for node in walk[1:]]
positive_embeddings = [lbann.Reshape(e, dims=f'1 {args.latent_dim}')
                       for e in positive_embeddings]
negative_embeddings = [lbann.Embedding(node,
                                       weights=embeddings,
                                       num_embeddings=num_graph_nodes,
                                       embedding_dim=args.latent_dim)
                       for node in negative_samples]
negative_embeddings = [lbann.Reshape(e, dims=f'1 {args.latent_dim}')
                       for e in negative_embeddings]
target_embeddings = lbann.Concatenation(positive_embeddings + negative_embeddings)

# Skip-Gram arhitecture
preds = lbann.MatMul(target_embeddings, source_embedding)
preds = lbann.Softmax(lbann.Reshape(preds, dims='-1'))
labels = \
    lbann.Concatenation(
        lbann.Constant(value=1, num_neurons=f'{walk_length-1}'),
        lbann.Constant(value=0, num_neurons=f'{num_negative_samples}')
    )
obj = lbann.CrossEntropy([preds, labels])

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

# Create LBANN objects
trainer = lbann.Trainer()
callbacks = [
    lbann.CallbackPrint(),
    lbann.CallbackTimer(),
    lbann.CallbackDumpWeights(basename='embeddings',
                              epoch_interval=args.num_epochs),
]
model = lbann.Model(args.mini_batch_size,
                    args.num_epochs,
                    layers=lbann.traverse_layer_graph(input_),
                    objective_function=obj,
                    callbacks=callbacks)
opt = lbann.SGD(learn_rate=0.025, momentum=0.9)

# Run LBANN
kwargs = lbann.contrib.args.get_scheduler_kwargs(args)
lbann.contrib.lc.launcher.run(trainer, model, reader, opt,
                              job_name=args.job_name,
                              experiment_dir=args.experiment_dir,
                              overwrite_script=True,
                              **kwargs)
