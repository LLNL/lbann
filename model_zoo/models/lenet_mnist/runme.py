#!/usr/bin/python

import sys
sys.path.insert(0, '../python')
import common

common.build_and_submit_slurm_script( 
   'model_lenet_mnist.prototext', 
   '../../data_readers/data_reader_mnist.prototext',
   '../../optimizers/opt_sgd.prototext' )
