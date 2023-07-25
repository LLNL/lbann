"""Driver script for training Transformer example."""
import argparse
import datetime
import math
import os
import os.path
import sys
from glob import glob

import lbann
import lbann.contrib.args

# Local imports
current_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)
import train
import evaluate_model
import utils.paths

# ----------------------------------------------
# Options
# ----------------------------------------------

# Command-line arguments
DAMPING_PARAM_NAMES = ["act", "err", "bn_act", "bn_err"]
def list2str(l):
    return ' '.join(l)

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
    '--num-layers', action='store', default=6, type=int,
    help='Number of encoder and decoder layers (default: 6)', metavar='NUM')

# KFAC configs
parser.add_argument("--kfac", dest="kfac", action="store_const",
                const=True, default=False,
                help="use the K-FAC optimizer (default: false)")
parser.add_argument("--disable-BN", dest="disBN", action="store_const",
                const=True, default=False,
                help="Disable KFAC for BN")

parser.add_argument("--poly-lr", dest="polyLR", action="store_const",
                const=True, default=False,
                help="Enable KFAC for BN")

parser.add_argument("--poly-decay", type=int, default=11,
                    help="decay in poly LR scheduler (default: 11)")

parser.add_argument("--dropout", dest="add_dropout", action="store_const",
                const=True, default=False,
                help="Add dropout after input")

parser.add_argument("--dropout-keep-val", type=float, default=0.8,
                help="Keep value of dropout layer after input (default: 0.8)")

parser.add_argument("--label-smoothing", type=float, default=0,
                help="label smoothing (default: 0)")

parser.add_argument("--mixup", type=float, default=0,
                    help="Data mixup (default: disabled)")

parser.add_argument("--momentum", type=float, default=2,
                    help="momentum in SGD overides optimizer  (default: 2(false))")

parser.add_argument("--enable-distribute-compute", dest="enable_distribute_compute", action="store_const",
                const=True, default=False,
                help="Enable distributed compute of precondition gradients")
parser.add_argument("--kfac-damping-warmup-steps", type=int, default=0,
                    help="the number of damping warmup steps")
parser.add_argument("--kfac-use-pi", dest="kfac_use_pi",
                    action="store_const",
                    const=True, default=False,
                    help="use the pi constant")

parser.add_argument("--kfac-sgd-mix", type=str, default="",
                        help="alogrithm will be switched to KFAC at first given epoch then alternate  (default: use KFAC for all epochs)")

parser.add_argument("--lr-list", type=str, default="",
                        help="change lr accroding to interval in --kfac-sgd-mix")
for n in DAMPING_PARAM_NAMES:
    parser.add_argument("--kfac-damping-{}".format(n), type=str, default="",
                        help="damping parameters for {}".format(n))
parser.add_argument("--kfac-update-interval-init", type=int, default=1,
                    help="the initial update interval of Kronecker factors")
parser.add_argument("--kfac-update-interval-target", type=int, default=1,
                    help="the target update interval of Kronecker factors")
parser.add_argument("--kfac-update-interval-steps", type=int, default=1,
                    help="the number of steps to interpolate -init and -target intervals")
parser.add_argument("--kfac-compute-interval-steps", type=int, default=1,
                    help="the number of steps after inverse matrices are calculated")
parser.add_argument("--use-eigen", dest="use_eigen",
                    action="store_const",
                    const=True, default=False)
# Debugging configs.
parser.add_argument("--print-matrix", dest="print_matrix",
                    action="store_const",
                    const=True, default=False)
parser.add_argument("--print-matrix-summary", dest="print_matrix_summary",
                    action="store_const",
                    const=True, default=False)
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
model_params = {
    'num_epochs': args.num_epochs,
    'embed_dim': args.embed_dim,
    'num_heads': args.num_attention_heads,
    'label_smoothing': label_smoothing,
    'num_layers' : args.num_layers,
}
script_params = lbann.contrib.args.get_scheduler_kwargs(args)
script_params['work_dir'] = work_dir
script_params['job_name'] = args.job_name
train_script = train.make_batch_script(
    trainer_params=trainer_params,
    model_params=model_params,
    script_params=script_params,
    args=args
)
train_script.run(overwrite=True)

# ----------------------------------------------
# Evaluate
# ----------------------------------------------
weights_prefix = glob(os.path.join(
    work_dir,
    'weights',
    'trainer0',
    f'*epoch.{args.num_epochs}*',
    'model0'
))[0]
evaluate_model.evaluate_transformer(weights_prefix)
