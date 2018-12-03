#!/usr/bin/python

import sys
sys.path.insert(0, '../python')
import common

common.build_and_submit_slurm_script( 
   'model_cosmoflow.prototext', 
   '../../data_readers/data_reader_synthetic_cosmoflow_128.prototext',
   '../../optimizers/opt_sgd.prototext' )
