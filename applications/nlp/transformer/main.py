import argparse
import datetime
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
import utils.paths

# ----------------------------------------------
# Options
# ----------------------------------------------

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

# Dataset properties
vocab_size = dataset.vocab_size()
sequence_length = dataset.sequence_length
pad_index = dataset.pad_index

# ----------------------------------------------
# Shared objects for training and validation
# ----------------------------------------------

# Directory for results
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
experiment_dir = os.path.join(
    utils.paths.root_dir(),
    'experiments',
    f'{timestamp}_{args.job_name}',
)

# Embedding weights
# Note: Glorot normal initialization
var = 2 / (args.embed_dim + vocab_size)
embedding_weights = lbann.Weights(
    name='embedding_weights',
    initializer=lbann.NormalInitializer(standard_deviation=math.sqrt(var)),
)

# Classifier weights
# TODO: Use embedding weights
classifier_matrix_weights = lbann.Weights(
    name='classifier_matrix_weights',
    initializer=lbann.NormalInitializer(standard_deviation=math.sqrt(var)),
)
classifier_bias_weights = lbann.Weights(
    name='classifier_bias_weights',
)

# Transformer model
model = model.Transformer(
    hidden_size=args.embed_dim,
    num_heads=args.num_attention_heads,
)

# ----------------------------------------------
# Layer graph for training
# ----------------------------------------------

# Input is two sequences of token IDs, separated by a pad token
input_ = lbann.Identity(lbann.Input())

# Get sequences of embedding vectors
# Note: Scale embeddings by sqrt(embed_dim).
# Note: Decoder input is shifted right, so first entry is pad token.
embeddings_tokens = lbann.Identity(lbann.Slice(
    input_,
    axis=0,
    slice_points=str_list([0, 2*sequence_length]),
))
embeddings = lbann.Embedding(
    embeddings_tokens,
    weights=embedding_weights,
    num_embeddings=vocab_size,
    embedding_dim=args.embed_dim,
    padding_idx=pad_index,
)
embeddings = lbann.WeightedSum(
    embeddings,
    scaling_factors=str(math.sqrt(args.embed_dim)),
)
embeddings_slice = lbann.Slice(
    embeddings,
    axis=0,
    slice_points=str_list([0, sequence_length, 2*sequence_length]),
)
encoder_input = lbann.Identity(embeddings_slice)
decoder_input = lbann.Identity(embeddings_slice)

# Apply transformer model
result = model(
    encoder_input, sequence_length,
    decoder_input, sequence_length,
)

# Use transformer decoder output to reconstruct decoder input
# TODO: Use embedding weights
preds = lbann.ChannelwiseFullyConnected(
    result,
    weights=[classifier_matrix_weights, classifier_bias_weights],
    output_channel_dims=[vocab_size],
)
preds = lbann.ChannelwiseSoftmax(preds)

# Cross entropy loss
# Note: Apply label smoothing.
label_tokens = lbann.Slice(
    input_,
    slice_points=str_list(range(sequence_length+1, 2*sequence_length+2)),
)
label_tokens = [lbann.Identity(label_tokens) for _ in range(sequence_length)]
labels = [lbann.OneHot(token, size=vocab_size) for token in label_tokens]
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

# ----------------------------------------------
# Data reader for training
# ----------------------------------------------

train_reader = lbann.reader_pb2.DataReader()
_reader = train_reader.reader.add()
_reader.name = 'python'
_reader.role = 'train'
_reader.percent_of_data_to_use = 1.0
_reader.python.module = 'dataset'
_reader.python.module_dir = current_dir
_reader.python.sample_function = 'get_train_sample'
_reader.python.num_samples_function = 'num_train_samples'
_reader.python.sample_dims_function = 'sample_dims'

# ----------------------------------------------
# Create batch script for training
# ----------------------------------------------

# Paths
train_experiment_dir = os.path.join(experiment_dir, 'train')
train_pb_file = os.path.join(train_experiment_dir, 'experiment.prototext')

# Create LBANN objects
trainer = lbann.Trainer()
metrics = []
callbacks = [
    lbann.CallbackPrint(),
    lbann.CallbackTimer(),
]
model = lbann.Model(
    args.mini_batch_size,
    args.num_epochs,
    layers=lbann.traverse_layer_graph(input_),
    objective_function=loss,
    metrics=metrics,
    callbacks=callbacks
)
opt = lbann.Adam(learn_rate=0.0004, beta1=0.9, beta2=0.98, eps=1e-9) # TODO: LR schedule

# Create batch script
kwargs = lbann.contrib.args.get_scheduler_kwargs(args)
train_script = lbann.contrib.lc.launcher.make_batch_script(
    job_name=args.job_name,
    work_dir=train_experiment_dir,
    **kwargs,
)
train_script.add_command('echo "Started training at $(date)"')
lbann.proto.save_prototext(
    train_pb_file,
    trainer=trainer,
    model=model,
    data_reader=train_reader,
    optimizer=opt
)
train_script.add_parallel_command([
    lbann.lbann_exe(),
    f'--prototext={train_pb_file}',
])
train_script.add_command('status=$?')
train_script.add_command('echo "Finished training at $(date)"')
train_script.add_command('exit ${status}')
train_script.write()
train_script.run(overwrite=True)
