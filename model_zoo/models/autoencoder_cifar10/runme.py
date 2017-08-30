#!/usr/bin/python

import sys
sys.path.insert(0, '../python')
import common

common.build_and_submit_slurm_script( 
    'model_autoencoder_cifar10.prototext', 
    '../../data_readers/data_reader_cifar10.prototext',
    '../../optimizers/opt_adam.prototext' ) 
