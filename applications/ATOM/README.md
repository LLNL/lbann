## Accelerating Therapeutics for Opportunities in Medicine (ATOM)

Models for training neural networks to suppor the [ATOM](https://atomscience.org) project

The train_atom_char_rnn.py script implements GRU-based recurrent model for generating new SMILES strings. 
Original neural network model and training hyperparameters are described in [MOSES benchmark](https://github.com/samadejacobs/moses/tree/master/moses/char_rnn). Please see LBANN documentations on how to install, build and run LBANN code. 

### How to install LBANN for this application
```bash
/bin/bash -c "$(curl -fsSL https://github.com/LLNL/lbann/raw/develop/applications/ATOM/build_lbann_atom_user.sh)"
```

### How to train
```bash
run python3 train_atom_char.rnn.py
```

Expected training output in LBANN (250K ZINC training dataset, on a single LLNL Pascal GPU) is shown:
```
--------------------------------------------------------------------------------
[0] Epoch : stats formated [tr/v/te] iter/epoch = [3907/0/0]
            global MB = [  64/   0/   0] global last MB = [  16  /   0  /   0  ]
             local MB = [  64/   0/   0]  local last MB = [  16+0/   0+0/   0+0]
--------------------------------------------------------------------------------
model0 (instance 0) training epoch 0 objective function : 0.438031
model0 (instance 0) training epoch 0 run time : 1009.55s
model0 (instance 0) training epoch 0 mini-batch time statistics : 0.257328s mean, 1.89938s max, 0.15177s min, 0.0331048s stdev
--------------------------------------------------------------------------------
[1] Epoch : stats formated [tr/v/te] iter/epoch = [3907/0/0]
            global MB = [  64/   0/   0] global last MB = [  16  /   0  /   0  ]
             local MB = [  64/   0/   0]  local last MB = [  16+0/   0+0/   0+0]
--------------------------------------------------------------------------------
model0 (instance 0) training epoch 1 objective function : 0.37321
model0 (instance 0) training epoch 1 run time : 1006.6s
model0 (instance 0) training epoch 1 mini-batch time statistics : 0.256573s mean, 0.912742s max, 0.158709s min, 0.0193512s stdev
```

### Inference and Sampling

1. Clone this version of [MOSES benchmark repository](https://github.com/samadejacobs/moses) and follow instructions for installation  
2. Inference using LBANN pretrained model parameters 

```bash

 python3 MOSES_DIR/scripts/run.py --model char_rnn  --n_samples NUM_SAMPLES \
                                  --lbann_weights_dir LBANN_WEIGHTS_DIR \
                                  --lbann_epoch_counts EPOCHS 

```

Command above will load pre_trained LBANN weights and biases from LBANN_WEIGHTS_DIR at a specified EPOCH counts, generate up to NUM_SAMPLES new molecules, and calculate metrics on the new molecules, some metrics relative to the test (validation) dataset.
