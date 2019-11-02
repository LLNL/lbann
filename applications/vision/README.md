# Example models for computer vision

This directory contains LBANN implementations of widely-used vision
models. They are intended to validate and benchmark LBANN's vision
functionality, and are also suitable as pedagogical tools for using
LBANN.

## LeNet

`lenet.py` trains a LeNet model on MNIST data. It is a simple script
intended to demonstrate LBANN's Python API. It calls helper functions
in `data/mnist/__init__.py` to download MNIST data and construct MNIST
data readers.

## ImageNet models

`alexnet.py`, `resnet.py`, and `densenet.py` are primarily used for
performance benchmarks and scaling studies. It uses LLNL-specific
features and the helper functions in `data/imagenet/__init__.py`
assume that the user is on an LLNL LC system and belongs to the
`brainusr` group.
