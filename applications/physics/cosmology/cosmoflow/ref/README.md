# PyTorch Reference Implementation of CosmoFlow

This directory contains a reference implementation of CosmoFlow in PyTorch as it is implemented for LBANN in the parent directory. This implementation uses `DistributedDataParallel` (DDP) to run on multiple GPUs. The included `batch.sh` script provides an example for training this model with DDP on Lassen.