#!/usr/bin/bash

python train_atom_char_rnn.py --job_name chembl_1_7m --embedding-dim 42 --num-embeddings 42 --data-path data/chembl/chembl_data_1_7m_train.npy --pad-index 40 --account baasic --partition pvis

