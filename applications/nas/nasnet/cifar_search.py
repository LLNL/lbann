import numpy as np
import sys
import os
import time
import lbann
import argparse
import lbann.contrib.args
import lbann.contrib.launcher
from os.path import join
import subprocess

current_file = os.path.realpath(__file__)
current_dir = os.path.dirname(current_file)
sys.path.insert(0, current_dir)
import cifar_networks

# ----------------------------------
# Command-line arguments
# ----------------------------------

desc = ('Micro search on CIFAR10 data using LBANN.')
parser = argparse.ArgumentParser(description=desc)
#lbann.contrib.args.add_scheduler_arguments(parser, 'denas_cifar10')

#NAS parameters
parser.add_argument(
    '--num-blocks', action='store', default=5, type=int,
    help='Number of blocks per cell (default: 5)')
parser.add_argument(
    '--n-ops', action='store', default=7, type=int,
    help='Number of operations (default: 7)')
parser.add_argument(
    '--n-cell', action='store', default=2, type=int,
    help='Number of cells (default: 2)')

parser.add_argument(
    '--use-ltfb', action='store_true', help='Use LTFB')

#Training (hyper) parameters
parser.add_argument(
    '--mini-batch-size', action='store', default=64, type=int,
    help='mini-batch size (default: 64)', metavar='NUM')
parser.add_argument(
    '--num-epochs', action='store', default=20, type=int,
    help='number of epochs (default: 20)', metavar='NUM')

#Compute (job) parameters
parser.add_argument(
    '--nodes', action='store', default=4, type=int,
    help='Num of compute nodes (default: 4)')
parser.add_argument(
    '--ppn', action='store', default=2, type=int,
    help='Processes per node (default: 2)')
parser.add_argument("--ppt", type=int, default=2)
parser.add_argument(
    '--job-name', action='store', default='denas_cifar10', type=str,
    help='scheduler job name (default: denas_cifar10)')

parser.add_argument(
    '--exp-dir', action='store', default='exp_cifar10', type=str,
    help='exp dir (default: exp_cifar10)')
lbann.contrib.args.add_optimizer_arguments(parser, default_learning_rate=0.1)
args = parser.parse_args()



if __name__ == "__main__":
    tag = 'ltfb' if args.use_ltfb else 'random'
    expd  = 'search-{}-{}-{}'.format('nasnet-micro-cifar10', tag, time.strftime("%Y%m%d-%H%M%S"))
    if not os.path.exists(expd):
        os.mkdir(expd)
    print('Experiment dir : {}'.format(expd))

    script = lbann.launcher.make_batch_script(nodes=args.nodes,
                                              procs_per_node=args.ppn,
                                              experiment_dir=expd)
    pop_size = int(args.nodes*args.ppn/args.ppt)

    cifar_networks.create_networks(expd,
                    args.num_epochs,
                    args.mini_batch_size,
                    pop_size,
                    use_ltfb=args.use_ltfb,
                    num_blocks=args.num_blocks,
                    num_ops=args.n_ops,
                    num_cells=args.n_cell)

    proto_file = os.path.join(script.work_dir,'experiment.prototext.trainer0')
    command = [
                lbann.lbann_exe(),
                f'--procs_per_trainer={args.ppt}',
                '--generate_multi_proto',
                f'--prototext={proto_file}']
    script.add_parallel_command(command)

    # Run script
    script.run(True)
