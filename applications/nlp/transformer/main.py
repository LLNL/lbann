import argparse
import datetime
import math
import os
import os.path
import sys

import lbann
import lbann.contrib.args

# Local imports
current_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)
import train
import evaluate
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
args = parser.parse_args()

# Hard-coded options
label_smoothing = 0.1

# Dataset properties
vocab_size = dataset.vocab_size()

# ----------------------------------------------
# Shared objects for training and validation
# ----------------------------------------------

# Directory for results
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
work_dir = os.path.join(
    utils.paths.root_dir(),
    'experiments',
    f'{timestamp}_{args.job_name}',
)

# Embedding weights
# Note: Glorot normal initialization
var = 2 / (args.embed_dim + vocab_size)
weights = {}
weights['embedding'] = lbann.Weights(
    name='embedding_weights',
    initializer=lbann.NormalInitializer(standard_deviation=math.sqrt(var)),
)

# Classifier weights
# TODO: Use embedding weights
weights['classifier_matrix'] = lbann.Weights(
    name='classifier_matrix_weights',
    initializer=lbann.NormalInitializer(standard_deviation=math.sqrt(var)),
)
weights['classifier_bias'] = lbann.Weights(
    name='classifier_bias_weights',
)

# Transformer model
transformer = model.Transformer(
    hidden_size=args.embed_dim,
    num_heads=args.num_attention_heads,
)

# ----------------------------------------------
# Train
# ----------------------------------------------

# Create work directory
os.makedirs(work_dir, exist_ok=True)



# Create batch script
train_params = {
    'mini_batch_size': args.mini_batch_size,
    'num_epochs': args.num_epochs,
    'embed_dim': args.embed_dim,
    'label_smoothing': label_smoothing,
}
batch_params = lbann.contrib.args.get_scheduler_kwargs(args)
batch_params['job_name'] = args.job_name
train_script = train.make_batch_script(
    transformer=transformer,
    weights=weights,
    work_dir=work_dir,
    train_params=train_params,
    batch_params=batch_params,
)
weights_prefix = os.path.join(
    work_dir,
    'weights',
    f'model0-epoch{args.num_epochs-1}',
)
train_script.add_command(
    f'# python3 {utils.paths.root_dir()}/transformer/evaluate.py {weights_prefix}'
)
train_script.run(overwrite=True)

# ----------------------------------------------
# Evaluate
# ----------------------------------------------

evaluate.evaluate_transformer(weights_prefix)
