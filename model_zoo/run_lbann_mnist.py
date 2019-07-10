#!/usr/bin/env python3
import argparse
import os.path
import google.protobuf.text_format as txtf
import lbann
from lbann import lbann_pb2

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
images = lbann.Identity(input,name='images')
labels = lbann.Identity(input)

# Layers
conv1 = lbann.Convolution(images,
                          num_dims = 2,
                          num_output_channels = 20,
                          conv_dims_i = 5,
                          conv_pads_i = 0,
                          conv_strides_i = 1,
                          has_bias = True)

pool1 = lbann.Pooling(conv1,
                      num_dims = 2,
                      pool_dims_i = 2,
                      pool_pads_i = 0,
                      pool_strides_i = 2,
                      pool_mode = "max")

conv2 = lbann.Convolution(pool1,
                          num_dims = 2,
                          num_output_channels = 50,
                          conv_dims_i = 5,
                          conv_pads_i = 0,
                          conv_strides_i = 1,
                          has_bias = True)

pool2 = lbann.Pooling(conv2,
                      num_dims = 2,
                      pool_dims_i = 2,
                      pool_pads_i = 0,
                      pool_strides_i = 2,
                      pool_mode = "max")

ip1 = lbann.FullyConnected(pool2,
                           num_neurons = 500,
                           has_bias = True)

relu1 = lbann.Relu(ip1)

ip2 = lbann.FullyConnected(relu1,
                           num_neurons = 10,
                           has_bias = True)

prob = lbann.Softmax(ip2)

# Loss function and accuracy
loss = lbann.CrossEntropy([prob, labels])

top1_accuracy = lbann.CategoricalAccuracy([prob, labels],name="bob")

img_dump_cb = lbann.CallbackDumpImageResults(cat_accuracy_layer="bob",
                                  image_layer="images",
                                  criterion=lbann_pb2.CallbackDumpImageResults.NOMATCH,
                                  interval=10)

# Setup model
mini_batch_size = 64
num_epochs = 2
model = lbann.Model(mini_batch_size,
                    num_epochs,
                    layers=lbann.traverse_layer_graph(input),
                    objective_function=loss,
                    metrics=[lbann.Metric(top1_accuracy, name='accuracy', unit='%')],
                    callbacks=[lbann.CallbackPrint(), lbann.CallbackTimer(),
                               lbann.CallbackSummary( dir = ".", batch_interval = 2,
                                                      mat_interval = 3),
                               img_dump_cb])

# Setup optimizer
opt = lbann.SGD(learn_rate=0.01, momentum=0.9)

# Load data reader from prototext
#model_zoo_dir = os.path.dirname(os.path.dirname(__file__))
model_zoo_dir = os.path.dirname(__file__)
print(model_zoo_dir)
data_reader_file = os.path.join(model_zoo_dir,
                                'data_readers',
                                'data_reader_mnist.prototext')
data_reader_proto = lbann.lbann_pb2.LbannPB()
with open(data_reader_file, 'r') as f:
    txtf.Merge(f.read(), data_reader_proto)
data_reader_proto = data_reader_proto.data_reader

# ----------------------------------
# Run experiment
# ----------------------------------
# Note: Use `lbann.contrib.lc.launcher.run` instead for optimized
# defaults on LC systems.

kwargs = {}
if args.partition: kwargs['partition'] = args.partition
if args.account: kwargs['account'] = args.account
lbann.run(model, data_reader_proto, opt,
          job_name='lbann_lenet',
          **kwargs)
