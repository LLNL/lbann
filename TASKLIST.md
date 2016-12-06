# Task List for LBANN: Livermore Big Artificial Neural Network Toolkit

## Core

###### Convolution
- [ ] GPU support for convolutional layers - performance tuning to minimize data transfer overhead
- [ ] GPU support for convolutional layers - optimizing the inter-kernel white space
- [ ] Improve performance of CPU based convolution layers

###### Misc
- [ ] Fix checkpoint and restore logic \(@adammoody\)
- [ ] Use GPU BLAS libraries for general matrix computations
- [ ] Test block based Elemental distribution versus element distribution

## New Features

### General
- [ ] Add support for batch normalization
- [ ] Add ImageNet support for mean subtraction
- [ ] Support partial training of network and fine-tuning
- [ ] Support for regression in addition to classification (http://deeplearning4j.org/linear-regression.html)

### Multi-model training

###### Pre-training + Fine-tuning
- [ ] Train multiple network layers (e.g. Greedy layer-wise)
- [ ] Freeze network layers, add new input and target layer, retrain last layers

###### Context Prediction
- [ ] Input layer: split image into paired patches
- [ ] Parallel train n-way Siamese network (i.e. tied weights)
- [ ] Target layer: multi-class problem

###### Text networks / embedding layers
- [ ] Input layer that selects row based on word one-hot encoding
- [ ] Backprop support for text embedding input layer

###### Bi-modal training
- [ ] Parallel forward passes
- [ ] Couple output of both forward passes
- [ ] Backprop combined result across image spoke

### ECP CANDLE
- [ ] Keras backend
- [ ] support for models defined by protofiles and weights saved in protobufs
- [ ] metrics / statistics class (look at imbalance python package)
- [ ] precision - recall curves
- [ ] confusion matrix
- [ ] novel data-parallel merging
- [ ] RNN / LSTM (layer / training) (Argonne will do)
