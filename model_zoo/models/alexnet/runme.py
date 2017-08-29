#!/usr/bin/python

import sys
sys.path.insert(0, '../python')
import common

common.build_and_submit_slurm_script( 
   'model_alexnet.prototext', 
   '../../data_readers/data_reader_imagenet.prototext',
   '../../optimizers/opt_adagrad.prototext' )
