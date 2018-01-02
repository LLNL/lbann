#!/usr/bin/python

import sys
sys.path.insert(0, '../python')
import common

common.build_and_submit_slurm_script( 
   'model_siamese.prototext', 
   'data_reader_imagenet_patches.prototext',
   '../../optimizers/opt_sgd.prototext' )
