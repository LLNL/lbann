import argparse
import lbann
import data.mnist
import lbann.contrib.args
import lbann.contrib.launcher
import lbann.util.amp

# ----------------------------------
# Command-line arguments
# ----------------------------------

desc = ('Train LeNet on MNIST data using LBANN.')
parser = argparse.ArgumentParser(description=desc)
lbann.contrib.args.add_scheduler_arguments(parser, 'lbann_lenet')
lbann.contrib.args.add_amp_arguments(parser)
args = parser.parse_args()

# ----------------------------------
# Construct layer graph
# ----------------------------------

# Input data
images = lbann.Input(data_field='samples')
labels = lbann.Input(data_field='labels')

# LeNet
x = lbann.Convolution(images,
                      num_dims = 2,
                      out_channels = 6,
                      groups = 1,
                      kernel_size = 5,
                      stride = 1,
                      dilation = 1,
                      has_bias = True)
x = lbann.Relu(x)
x = lbann.Pooling(x,
                  num_dims = 2,
                  pool_dims_i = 2,
                  pool_strides_i = 2,
                  pool_mode = "max")
x = lbann.Convolution(x,
                      num_dims = 2,
                      out_channels = 16,
                      groups = 1,
                      kernel_size = 5,
                      stride = 1,
                      dilation = 1,
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
                    layers=lbann.traverse_layer_graph([images, labels]),
                    objective_function=loss,
                    metrics=[lbann.Metric(acc, name='accuracy', unit='%')],
                    callbacks=[lbann.CallbackPrintModelDescription(),
                               lbann.CallbackPrint(),
                               lbann.CallbackTimer()])

lbann.util.amp.enable_amp(model, args)

# Setup optimizer
opt = lbann.SGD(learn_rate=0.01, momentum=0.9)

# Setup data reader
data_reader = data.mnist.make_data_reader()

# Setup trainer
trainer = lbann.Trainer(mini_batch_size=mini_batch_size)

# ----------------------------------
# Run experiment
# ----------------------------------
kwargs = lbann.contrib.args.get_scheduler_kwargs(args)
lbann.contrib.launcher.run(trainer, model, data_reader, opt,
                           job_name=args.job_name,
                           **kwargs)
