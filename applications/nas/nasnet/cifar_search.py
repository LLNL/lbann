# NASNet Search Space https://arxiv.org/pdf/1707.07012.pdf
# code modified from DARTS https://github.com/quark0/darts
import numpy as np
import sys
import os
import time
from collections import namedtuple
import lbann
import lbann.models
import lbann.models.resnet
import data.cifar10
from search import micro_encoding
import argparse
import lbann.contrib.args
import lbann.contrib.launcher
from os.path import join
import subprocess

sys.path.insert(0, os.getenv('PWD'))
import search.model as cifar


Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
Genotype_norm = namedtuple('Genotype', 'normal normal_concat')
Genotype_redu = namedtuple('Genotype', 'reduce reduce_concat')

# what you want to search should be defined here and in micro_operations
PRIMITIVES = [
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5',
    'sep_conv_7x7',
    'conv_7x1_1x7',
]
'''

PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5',
]
'''

# ----------------------------------
# Command-line arguments
# ----------------------------------

desc = ('Micro search on CIFAR10 data using LBANN.')
parser = argparse.ArgumentParser(description=desc)
#lbann.contrib.args.add_scheduler_arguments(parser)

#NAS parameters
parser.add_argument(
    '--pop-size', action='store', default=16, type=int,
    help='Population size (default: 16)')
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
    '--num-epochs', action='store', default=200, type=int,
    help='number of epochs (default: 10)', metavar='NUM')

#Compute (job) parameters
parser.add_argument(
    '--nodes', action='store', default=4, type=int,
    help='Num of compute nodes (default: 4)')
parser.add_argument(
    '--ppn', action='store', default=4, type=int,
    help='Processes per node (default: 4)')
parser.add_argument("--ppt", type=int, default=2) 
parser.add_argument(
    '--job-name', action='store', default='denas_cifar10', type=str,
    help='scheduler job name (default: denas_cifar10)')

parser.add_argument(
    '--exp-dir', action='store', default='exp_cifar10', type=str,
    help='exp dir (default: exp_cifar10)')
lbann.contrib.args.add_optimizer_arguments(parser, default_learning_rate=0.1)
args = parser.parse_args()


def generate(expd):
    # design to debug the encoding scheme
    seed = 0
    np.random.seed(seed)
    pop_size = int(args.nodes*args.ppn/args.ppt)
    #B, n_ops, n_cell = 5, 7, 2
    B, n_ops, n_cell = args.num_blocks, args.n_ops, args.n_cell
    networks = []
    network_id = 0
        
    # Setup data reader
    data_reader = data.cifar10.make_data_reader(num_classes=10)
       
    while len(networks) < pop_size:
        bit_string = []
        for c in range(n_cell):
            for b in range(B):
                bit_string += [np.random.randint(n_ops),
                               np.random.randint(b + 2),
                               np.random.randint(n_ops),
                               np.random.randint(b + 2)
                               ]

        genome = micro_encoding.convert(bit_string)
        # check against evaluated networks in case of duplicates
        doTrain = True
        for network in networks:
            if micro_encoding.compare(genome, network):
                doTrain = False
                break

        if doTrain:
            genotype = micro_encoding.decode(genome)
            print("Newtwork id, bitstring, genome, genotype ", network_id, bit_string, genome, genotype)
            mymodel = cifar.NetworkCIFAR(16, 10, 8, False, genotype)
            networks.append(genome)

            images = lbann.Input(data_field='samples')
            labels = lbann.Input(data_field='labels')

            preds,_ = mymodel(images)
            probs = lbann.Softmax(preds)
            cross_entropy = lbann.CrossEntropy(probs, labels)
            top1 = lbann.CategoricalAccuracy(probs, labels)

            obj = lbann.ObjectiveFunction([cross_entropy])
    

            metrics = lbann.Metric(top1, name='accuracy', unit='%')

            callbacks = [lbann.CallbackPrint(),
                         lbann.CallbackTimer()]


            model = lbann.Model(epochs=args.num_epochs,
                       layers=[images,labels],
                       objective_function=obj,
                       metrics=metrics,
                       callbacks=callbacks)

            opt = lbann.contrib.args.create_optimizer(args)

            # Setup trainer
            trainer = lbann.Trainer(mini_batch_size=args.mini_batch_size)
            
            if args.use_ltfb:
                print("Using LTFB ")
                SGD = lbann.BatchedIterativeOptimizer
                RPE = lbann.RandomPairwiseExchange
                ES = lbann.RandomPairwiseExchange.ExchangeStrategy(strategy='checkpoint_binary')
                #            checkpoint_dir=os.getcwd())
                metalearning = RPE(
                   metric_strategies={'accuracy': RPE.MetricStrategy.HIGHER_IS_BETTER},
                   exchange_strategy=ES)  
                ltfb = lbann.LTFB("ltfb",
                              metalearning=metalearning,
                              local_algo=SGD("local sgd", num_iterations=625), 
                              #local_algo=SGD("local sgd", num_iterations=62),
                              metalearning_steps=args.num_epochs)

                trainer = lbann.Trainer(mini_batch_size=args.mini_batch_size,
                                    training_algo=ltfb)

            
    
            # Run LBANN
            try:
                lbann.contrib.launcher.run(trainer=trainer,
                    model=model,
                    time_limit=360,
                    account='hpcml',
                    experiment_dir=expd,
                    nodes=args.nodes,
                    procs_per_node=args.ppn,
                    data_reader=data_reader,
                    optimizer=opt,
                    setup_only=True,
                    proto_file_name="experiment.prototext.trainer"+str(network_id),
                    lbann_args=f"--generate_multi_proto --procs_per_trainer={args.ppt}",
                    job_name=args.job_name)
            except:
                pass
            
            network_id += 1


if __name__ == "__main__":
    tag = 'ltfb' if args.use_ltfb else 'random'
    expd  = 'search-{}-{}-{}'.format('nasnet-micro-cifar10', tag, time.strftime("%Y%m%d-%H%M%S"))
    if not os.path.exists(expd):
        os.mkdir(expd)
    print('Experiment dir : {}'.format(expd))
    generate(expd)
    
    
    
                     
