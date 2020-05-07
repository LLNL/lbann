import argparse
import lbann
import data.mnist

# ----------------------------------
# Command-line arguments
# ----------------------------------

desc = ('Train LeNet on MNIST data using LBANN.')
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
input_ = lbann.Input()
images = lbann.Identity(input_)
labels = lbann.Identity(input_)

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
loss = lbann.CrossEntropy(probs, labels)
acc = lbann.CategoricalAccuracy(probs, labels)

# ----------------------------------
# Setup experiment
# ----------------------------------

# Setup model
mini_batch_size = 64
num_epochs = 20
model = lbann.Model(num_epochs,
                    layers=lbann.traverse_layer_graph(input_),
                    objective_function=loss,
                    metrics=[lbann.Metric(acc, name='accuracy', unit='%')],
                    callbacks=[lbann.CallbackPrintModelDescription(),
                               lbann.CallbackPrint(),
                               lbann.CallbackTimer()])

# Setup optimizer
opt = lbann.SGD(learn_rate=0.01, momentum=0.9)

# Setup data reader
data_reader = data.mnist.make_data_reader()

# Setup trainer
trainer = lbann.Trainer(mini_batch_size=mini_batch_size)

# ----------------------------------
# Run experiment
# ----------------------------------
# Note: Use `lbann.contrib.launcher.run` instead for optimized
# defaults.

kwargs = {}
if args.partition: kwargs['partition'] = args.partition
if args.account: kwargs['account'] = args.account
lbann.run(trainer, model, data_reader, opt, **kwargs)
