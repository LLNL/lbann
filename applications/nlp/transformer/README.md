# Distributed Transformer Training Examples with LBANN

This folder contains sample training runs for different types of transformers.

LBANN will train the transformers on anything from a single node to a supercomputer.

The driver scripts are:
* `train_transformer_translation.py`: Trains a Transformer (Encoder-Decoder)
  neural network on a source-target sequence task. A WMT-16 dataset reader is
  provided as a sample translation task.
* `pretrain_gpt.py`: Pre-trains a GPT-3-like transformer on a causal language
  modeling task without masking. The Pile dataset reader is provided as an
  example.


## Contents

Apart from the driver scripts, the folder contains the following modules:
* `pytorch-reference`: A folder containing reference PyTorch implementations
  of the models (without distributed training support).
* `datasets/*`: Multiple real and synthetic datasets that can be used by the
  drivers with the `--dataset` flag.
* `dataset_utils.py`: Utilities to query and load datasets in the folder.
* `evaluate_translation_bleu.py`: Evaluate the results of the `train_transformer_translation`
  driver script with a BLEU score. Flags passed to the script must be the same as
  the ones given for training.
* `modeling.py`: Contains LBANN transformer model setup functionality.
* `parallelism.py`: Contains different parallelization strategies for running
  distributed transformer training. See `--help` for more information about
  how to use the strategies.
* `trainer.py`: Contains LBANN training setup and optimizer functionality.
