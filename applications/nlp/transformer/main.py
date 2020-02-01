import argparse
import math
import os.path
import sys

import lbann
import lbann.contrib.lc.launcher
import lbann.contrib.args
from lbann.util import str_list

# Local imports
current_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)
import dataset
import model

# ----------------------------------
# Options
# ----------------------------------

# Command-line arguments
parser = argparse.ArgumentParser()
lbann.contrib.args.add_scheduler_arguments(parser)
parser.add_argument(
    '--job-name', action='store', default='lbann_transformer', type=str,
    help='job name', metavar='NAME')
parser.add_argument(
    '--mini-batch-size', action='store', default=256, type=int,
    help='mini-batch size (default: 256)', metavar='NUM')
parser.add_argument(
    '--num-epochs', action='store', default=20, type=int,
    help='number of epochs (default: 20)', metavar='NUM')
parser.add_argument(
    '--num-attention-heads', action='store', default=8, type=int,
    help='number of parallel attention layers (default: 8)', metavar='NUM')
parser.add_argument(
    '--embed-dim', action='store', default=512, type=int,
    help='embedding space dimensions (default: 512)', metavar='NUM')
parser.add_argument(
    '--label-smoothing', action='store', default=0.1, type=float,
    help='label smoothing (default: 0.1)', metavar='VAL')
args = parser.parse_args()

# ----------------------------------
# Construct layer graph
# ----------------------------------

# Dataset properties
vocab_size = dataset.corpus.vocab_size
sequence_length = dataset.sample_dims()[0]

# Input is a sequence of token IDs
input_ = lbann.Identity(lbann.Input())
input_slice = lbann.Slice(input_,
                          slice_points=str_list(range(sequence_length+1)))
tokens_list = [lbann.Identity(input_slice) for _ in range(sequence_length)]

# Get sequence of embedding vectors
# Note: Inputs to transformer decoder are shifted right
# Note: Embeddings are scaled by sqrt(embed_dim)
embeddings = lbann.Embedding(
    lbann.Concatenation(lbann.Constant(value=-1, num_neurons='1'), input_),
    num_embeddings=vocab_size,
    embedding_dim=args.embed_dim
)
embeddings = lbann.WeightedSum(
    embeddings,
    scaling_factors=str(math.sqrt(args.embed_dim)),
)
source_embeddings = lbann.Identity(lbann.Slice(
    embeddings,
    axis=0,
    slice_points=str_list([1, sequence_length+1]),
))
target_embeddings = lbann.Identity(lbann.Slice(
    embeddings,
    axis=0,
    slice_points=str_list([0, sequence_length]),
))

# Apply transformer model
model = model.Transformer(
    hidden_size=args.embed_dim,
    num_heads=args.num_attention_heads,
    dropout=0,  # TODO: Restore dropout
)
result = model(
    source_embeddings, sequence_length,
    target_embeddings, sequence_length,
)

# Use transformer decoder output to reconstruct input sequence
preds = lbann.ChannelwiseFullyConnected(
    result,
    output_channel_dims=[vocab_size],
)
preds = lbann.ChannelwiseSoftmax(preds)

# Cross entropy loss
# Note: Apply label smoothing
labels = [lbann.OneHot(token, size=vocab_size) for token in tokens_list]
labels = lbann.Concatenation(
    [lbann.Reshape(label, dims=str_list([1, vocab_size])) for label in labels],
    axis=0,
)
if args.label_smoothing > 0:
    uniform_labels = lbann.Constant(
        value=1/vocab_size,
        num_neurons=str_list([sequence_length, vocab_size])
    )
    labels = lbann.WeightedSum(
        labels,
        uniform_labels,
        scaling_factors=str_list([1-args.label_smoothing, args.label_smoothing]),
    )
loss = lbann.CrossEntropy(preds, labels)

# ----------------------------------
# Create data reader
# ----------------------------------

reader = lbann.reader_pb2.DataReader()
_reader = reader.reader.add()
_reader.name = 'python'
_reader.role = 'train'
_reader.percent_of_data_to_use = 1.0
_reader.python.module = 'dataset'
_reader.python.module_dir = current_dir
_reader.python.sample_function = 'get_sample'
_reader.python.num_samples_function = 'num_samples'
_reader.python.sample_dims_function = 'sample_dims'

# ----------------------------------
# Run LBANN
# ----------------------------------

# Create LBANN objects
trainer = lbann.Trainer()
metrics = []
callbacks = [lbann.CallbackPrint(),
             lbann.CallbackTimer()]
model = lbann.Model(args.mini_batch_size,
                    args.num_epochs,
                    layers=lbann.traverse_layer_graph(input_),
                    objective_function=loss,
                    metrics=metrics,
                    callbacks=callbacks)
opt = lbann.Adam(learn_rate=0.0004, beta1=0.9, beta2=0.98, eps=1e-9) # TODO: LR schedule

# Run LBANN
kwargs = lbann.contrib.args.get_scheduler_kwargs(args)
lbann.contrib.lc.launcher.run(trainer, model, reader, opt,
                              job_name=args.job_name,
                              **kwargs)
