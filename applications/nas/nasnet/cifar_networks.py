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
from search import micro_encoding
from os.path import join
import data.cifar10

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

def generate_genomes(pop_size,
                     num_blocks=5,
                     num_ops=7,
                     num_cells=2):
    seed = 0
    np.random.seed(seed)
    B, n_ops, n_cell = num_blocks, num_ops, num_cells
    networks = []
    genotypes = []
    network_id = 0
        
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
            networks.append(genome)
            genotypes.append(genotype)
            network_id +=1

    return genotypes


def create_networks(exp_dir,
                    num_epochs,
                    mini_batch_size,
                    pop_size,
                    use_ltfb=False,
                    num_blocks=5,
                    num_ops=7,
                    num_cells=2,
                    ):
        trainer_id = 0
        # Setup shared data reader and optimizer
        reader = data.cifar10.make_data_reader(num_classes=10)
        opt = lbann.Adam(learn_rate=0.0002,beta1=0.9,beta2=0.99,eps=1e-8) 
        genotypes = generate_genomes(pop_size,num_blocks,num_ops,num_cells)
        for g in genotypes:
            mymodel = cifar.NetworkCIFAR(16, 10, 8, False, g)

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


            model = lbann.Model(epochs=num_epochs,
                       layers=[images,labels],
                       objective_function=obj,
                       metrics=metrics,
                       callbacks=callbacks)


            # Setup trainer
            trainer = lbann.Trainer(mini_batch_size=mini_batch_size)
            
            if use_ltfb:
                print("Using LTFB ")
                SGD = lbann.BatchedIterativeOptimizer
                RPE = lbann.RandomPairwiseExchange
                ES = lbann.RandomPairwiseExchange.ExchangeStrategy(strategy='checkpoint_binary')
                metalearning = RPE(
                   metric_strategies={'accuracy': RPE.MetricStrategy.HIGHER_IS_BETTER},
                   exchange_strategy=ES)  
                ltfb = lbann.LTFB("ltfb",
                              metalearning=metalearning,
                              local_algo=SGD("local sgd", num_iterations=625), 
                              metalearning_steps=num_epochs)

                trainer = lbann.Trainer(mini_batch_size=mini_batch_size,
                                    training_algo=ltfb)

             # Export Protobuf file
            lbann.proto.save_prototext(
               os.path.join(exp_dir, f'experiment.prototext.trainer{trainer_id}'),
               model=model,
               optimizer=opt,
               data_reader=reader,
               trainer=trainer)
            
            trainer_id +=1
    
        return trainer, model, reader, opt 
    

