#!/usr/bin/bash

python train_atom_char_rnn.py --job_name chembl_mw_filter_250_450 --embedding-dim 40 --num-embeddings 40 --data-path data/chembl_mw_filter_250_450/train.npy --pad-index 38 --account baasic --partition pbatch

