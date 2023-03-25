"""Driver script for training Transformer example."""
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
import train_infer as train
import evaluate
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
    '--dkv', action='store', default=0, type=int,
    help='D_kv dimension per head (default: 0 == embed_dim/num_heads )', metavar='NUM')

parser.add_argument(
    '--encoder-layers', action='store', default=6, type=int,
    help='Number of encoder layers (default: 6)', metavar='NUM')



parser.add_argument(
    '--decoder-layers', action='store', default=6, type=int,
    help='Number of decoder layers (default: 6)', metavar='NUM')

parser.add_argument(
    '--filter-size', action='store', default=2048, type=int,
    help='Number of neurons Dff (default: 2048)', metavar='NUM')

parser.add_argument(
    '--branches', action='store', default=0, type=int,
    help='Number of Branches 0 means DP (default: 0)', metavar='NUM')

parser.add_argument(
    '--subgraph-topology', action='store', default=0, type=int,
    help='Stategy for topology aware subgraph parallelism (default: 0) ', metavar='NUM')

parser.add_argument(
    '--subgraph-parent-resources', action='store', default=0, type=int,
    help='NUmber of resources for parent/common layers (corresponds to use all ranks) (default: 0) ', metavar='NUM')

parser.add_argument('--enable-allsubgraph', dest='ENABLE_ALLSUBGRAPH', action='store_true',
                        help='Enable subgraph parallelism for common layers in Encoder')
parser.add_argument('--enable-concat', dest='ENABLE_Concat', action='store_true',
                        help='Apply concat operation after each encoder layers when AllSubgraph variable is given')

args = parser.parse_args()




# Hard-coded options
label_smoothing = 0.1

# ----------------------------------------------
# Work directory
# ----------------------------------------------

timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
work_dir = os.path.join(
    utils.paths.root_dir(),
    'experiments',
    f'{timestamp}_{args.job_name}',
)
os.makedirs(work_dir, exist_ok=True)

# ----------------------------------------------
# Train
# ----------------------------------------------

# Create batch script
trainer_params = {
    'mini_batch_size': args.mini_batch_size,
}

if(args.dkv==0):
    d_kv = None
else:
    d_kv = args.dkv
model_params = {
    'num_epochs': args.num_epochs,
    'embed_dim': args.embed_dim,
    'num_heads': args.num_attention_heads,
    'label_smoothing': label_smoothing,
    'branches': args.branches,
    'subgraph_topology':args.subgraph_topology,
    'subgraph_num_common_resources': args.subgraph_parent_resources,
    'num_encoder_layers':args.encoder_layers,
    'num_decoder_layers':args.decoder_layers,
    'filter_size':args.filter_size,
    'd_kv': d_kv,
    'ENABLE_ALLSUBGRAPH': args.ENABLE_ALLSUBGRAPH,
    'ENABLE_Concat': args.ENABLE_Concat
}
script_params = lbann.contrib.args.get_scheduler_kwargs(args)
script_params['work_dir'] = work_dir
script_params['job_name'] = args.job_name
#script_params['mini_batch_size'] = args.mini_batch_size
#print("script params",trainer_params['mini_batch_size'])
train.make_batch_script(
    trainer_params=trainer_params,
    model_params=model_params,
    script_params=script_params,
)
# weights_prefix = os.path.join(
#     work_dir,
#     'weights',
#     f'model0-epoch{args.num_epochs-1}',
# )
# train_script.add_command(
#     f'# python3 {utils.paths.root_dir()}/transformer/evaluate.py {weights_prefix}'
# )
# train_script.run(overwrite=True)

# ----------------------------------------------
# Evaluate
# ----------------------------------------------
#evaluate.evaluate_transformer(weights_prefix)
