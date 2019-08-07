#!/usr/bin/env python3
import argparse
import os.path
import lbann

# ----------------------------------
# Command-line arguments
# ----------------------------------

desc = ('Construct and run LeNet on MNIST data. '
        'Running the experiment is only supported on LC systems.')
parser = argparse.ArgumentParser(description=desc)
parser.add_argument(
    '--partition', action='store', type=str,
    help='scheduler partition', metavar='NAME')
parser.add_argument(
    '--account', action='store', type=str,
    help='scheduler account', metavar='NAME')
args = parser.parse_args()

# ----------------------------------
# Construct layer graph
# ----------------------------------

# Input data
input = lbann.Input()
images = lbann.Identity(input)
labels = lbann.Identity(input)

# LeNet
x = lbann.Convolution(images,
                      num_dims = 2,
                      num_output_channels = 6,
                      num_groups = 1,
                      conv_dims_i = 5,
                      conv_strides_i = 1,
                      conv_dilations_i = 1,
                      has_bias = True)
x = lbann.Relu(x)
x = lbann.Pooling(x,
                  num_dims = 2,
                  pool_dims_i = 2,
                  pool_strides_i = 2,
                  pool_mode = "max")
x = lbann.Convolution(x,
                      num_dims = 2,
                      num_output_channels = 16,
                      num_groups = 1,
                      conv_dims_i = 5,
                      conv_strides_i = 1,
                      conv_dilations_i = 1,
                      has_bias = True)
x = lbann.Relu(x)
x = lbann.Pooling(x,
                  num_dims = 2,
                  pool_dims_i = 2,
                  pool_strides_i = 2,
                  pool_mode = "max")
x = lbann.FullyConnected(x, num_neurons = 120, has_bias = True)
x = lbann.Relu(x)
x = lbann.FullyConnected(x, num_neurons = 84, has_bias = True)
x = lbann.Relu(x)
x = lbann.FullyConnected(x, num_neurons = 10, has_bias = True)
probs = lbann.Softmax(x)

# Loss function and accuracy
loss = lbann.CrossEntropy([probs, labels])
acc = lbann.CategoricalAccuracy([probs, labels])

# ----------------------------------
# Setup experiment
# ----------------------------------

# Setup model
mini_batch_size = 64
num_epochs = 4
model = lbann.Model(mini_batch_size,
                    num_epochs,
                    layers=lbann.traverse_layer_graph(input),
                    objective_function=loss,
                    metrics=[lbann.Metric(acc, name='accuracy', unit='%')],
                    callbacks=[lbann.CallbackPrint(), lbann.CallbackTimer()])

# Setup optimizer
opt = lbann.SGD(learn_rate=0.01, momentum=0.9)

# Setup DataReader
message = lbann.lbann_pb2.DataReader()

# Train set data reader
data_reader = message.reader.add()
data_reader.name = 'mnist'
data_reader.role = 'train'
data_reader.shuffle = True
data_reader.data_filedir = '/p/lscratchh/brainusr/datasets/MNIST'
data_reader.data_filename = 'train-images-idx3-ubyte'
data_reader.label_filename = 'train-labels-idx1-ubyte'
data_reader.validation_percent = 0.1
data_reader.absolute_sample_count = 0
data_reader.percent_of_data_to_use = 1.0
transforms = data_reader.transforms.add()
transforms.scale.scale = 0.003921568627
transforms.AdjustBrightness.factor = 0.999999
transforms.AdjustSaturation.factor = 0.999999

# Test set data reader
data_reader = message.reader.add()
data_reader.name = 'mnist'
data_reader.role = 'test'
data_reader.shuffle = True
data_reader.data_filedir = '/p/lscratchh/brainusr/datasets/MNIST'
data_reader.data_filename = 't10k-images-idx3-ubyte'
data_reader.label_filename = 't10k-labels-idx1-ubyte'
data_reader.absolute_sample_count = 0
data_reader.percent_of_data_to_use = 1.0
transforms = data_reader.transforms.add()
transforms.scale.scale = 0.003921568627


# ----------------------------------
# Run experiment
# ----------------------------------
# Note: Use `lbann.contrib.lc.launcher.run` instead for optimized
# defaults on LC systems.

kwargs = {}
if args.partition: kwargs['partition'] = args.partition
if args.account: kwargs['account'] = args.account
lbann.run(model, message, opt,
          job_name='lbann_try',
          **kwargs)
