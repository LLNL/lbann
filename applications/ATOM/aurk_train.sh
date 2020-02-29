#!/usr/bin/bash

python train_atom_char_rnn.py --job_name aurk --embedding-dim 39 --num-embeddings 39 --data-path data/aurk_base/train.npy --pad-index 37 --account baasic --partition pbatch

