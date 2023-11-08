# This file is part of LBANN: Livermore Big Artificial Neural Network
# Toolkit. For details, see http://software.llnl.gov/LBANN or
# https://github.com/LLNL/LBANN.
#
# Licensed under the Apache License, Version 2.0 (the "Licensee"); you
# may not use this file except in compliance with the License.  You may
# obtain a copy of the License at:
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the license.
#
# vae_mnist.py - An LBANN implementation of MNIST VAE in Doersch's
# autoencoder tutorial
#
# See https://github.com/cdoersch/vae_tutorial/blob/master/mnist_vae.prototxt
#
################################################################################

import os.path
import google.protobuf.text_format as txtf
import sys
import argparse
import lbann
import lbann.models
import lbann.contrib.args
import lbann.contrib.models.wide_resnet
import lbann.contrib.launcher

# Get relative path to data
current_file = os.path.realpath(__file__)
current_dir = os.path.dirname(current_file)
sys.path.insert(0, os.path.join(os.path.dirname(current_dir), 'data'))
import mnist

# Command-line arguments
desc = ('An LBANN implementation of MNIST VAE in Doersch\'s autoencoder tutorial. '
        'Running the experiment is only supported on LC systems.')
parser = argparse.ArgumentParser(description=desc)
lbann.contrib.args.add_scheduler_arguments(parser, 'lbann_vae_mnist')
parser.add_argument(
    '--mini-batch-size', action='store', default=100, type=int,
    help='mini-batch size (default: 100)', metavar='NUM')
parser.add_argument(
    '--num-epochs', action='store', default=50, type=int,
    help='number of epochs (default: 50)', metavar='NUM')
parser.add_argument(
    '--random-seed', action='store', default=0, type=int,
    help='random seed for LBANN RNGs', metavar='NUM')
parser.add_argument(
    '--data-reader', action='store', default='default', type=str,
    help='Data reader options: \"numpy_npz_int16\", or \"mnist\" (default: data_reader_mnist.prototext)')
lbann.contrib.args.add_optimizer_arguments(parser, default_learning_rate=0.1)
args = parser.parse_args()

# Start of layers

# Construct layer graph
input_ = lbann.Input(name='data')
image = lbann.Split(input_, name='image')
dummy = lbann.Dummy(input_, name='dummy')

# Encoder
encode1 = lbann.FullyConnected(image,
                               name="encode1",
                               num_neurons=1000,
                               has_bias=True)

encode1neuron = lbann.Relu(encode1, name="encode1neuron")

encode2 = lbann.FullyConnected(encode1neuron,
                               name="encode2",
                               num_neurons=500,
                               has_bias=True)

encode2neuron = lbann.Relu(encode2, name="encode2neuron")

encode3 = lbann.FullyConnected(encode2neuron,
                               name="encode3",
                               num_neurons=250,
                               has_bias=True)

encode3neuron = lbann.Relu(encode3, name="encode3neuron")

# Latent space
mu = lbann.FullyConnected(encode3neuron,
                          name="mu",
                          num_neurons=30,
                          has_bias=True)

logsd = lbann.FullyConnected(encode3,
                             name="logsd",
                             num_neurons=30,
                             has_bias=True)

# KL divergence
sd = lbann.Exp(logsd, name="sd")

var = lbann.Square(sd, name="var")

meansq = lbann.Square(mu, name="meansq")

kldiv_plus_half = lbann.WeightedSum([meansq, var, logsd],
                                    name="kldiv_plus_half",
                                    scaling_factors=[0.5, 0.5, -1])

kldiv_full = lbann.Rsqrt(kldiv_plus_half, name="kldiv_full")

kldiv = lbann.Reduction(kldiv_full, name="kldiv", mode="sum")

# Generate sample
noise = lbann.Gaussian(name="noise", mean=0, stdev=1, hint_layer=mu)

sdnoise = lbann.Hadamard([noise, sd], name="sdnoise")

sample = lbann.Add([mu, sdnoise], name="sample")

# Decoder
decode4 = lbann.FullyConnected(sample,
                               name="decode4",
                               has_bias=True,
                               hint_layer=encode3)

decode4neuron = lbann.Relu(decode4, name="decode4neuron")

decode3 = lbann.FullyConnected(decode4neuron,
                               name="decode3",
                               has_bias=True,
                               hint_layer=encode2)

decode3neuron = lbann.Relu(decode3, name="decode3neuron")

decode2 = lbann.FullyConnected(decode3neuron,
                               name="decode2",
                               has_bias=True,
                               hint_layer=encode1)

decode2neuron = lbann.Relu(decode2, name="decode2neuron")

decode1 = lbann.FullyConnected(decode2neuron,
                               name="decode1",
                               has_bias=True,
                               hint_layer=image)

# Reconstruction error
reconstruction = lbann.Sigmoid(decode1, name="reconstruction")

bin_cross_entropy = lbann.SigmoidBinaryCrossEntropy([decode1, image],
                                                    name="bin_cross_entropy")

bin_cross_entropy_sum = lbann.Reduction(bin_cross_entropy,
                                        name="bin_cross_entropy_sum",
                                        mode="sum")

mean_squared_error = lbann.MeanSquaredError([reconstruction, image],
                                            name="mean_squared_error")

layer_list = list(lbann.traverse_layer_graph(input_))

# Set up objective function
layer_term1 = lbann.LayerTerm(bin_cross_entropy)
layer_term2 = lbann.LayerTerm(kldiv)
l2_reg = lbann.L2WeightRegularization(scale=0.0005)
obj = lbann.ObjectiveFunction([layer_term1, layer_term2, l2_reg])

# Metrics
metrics = [lbann.Metric(mean_squared_error, name="mean squared error")]

# Callbacks
callbacks = [lbann.CallbackPrint(),
             lbann.CallbackTimer(),
             lbann.CallbackSaveImages(layers="image reconstruction",
                                       image_format="jpg")]

# Setup Model
model = lbann.Model(args.num_epochs,
                    layers=layer_list,
                    objective_function=obj,
                    metrics=metrics,
                    callbacks=callbacks,
                    summary_dir=".")

# Setup optimizer
opt = lbann.contrib.args.create_optimizer(args)

# Setup data reader
data_reader_prefix = 'data_reader_mnist'
if args.data_reader == "default" or args.data_reader == "mnist":
  data_reader_file = data_reader_prefix + '.prototext'
elif args.data_reader == "numpy_npz_int16":
  data_reader_file = data_reader_prefix + '_numpy_npz_int16.prototext'
else:
  raise InvalidOption('Data reader selection \"' + args.data_reader + '\" is invalid. Use \"numpy_npz_int16\", or \"mnist\". Default is data_reader_mnist.prototext.')

data_reader = mnist.make_data_reader(data_reader_file)


# Setup trainer
trainer = lbann.Trainer(mini_batch_size=args.mini_batch_size)

# Run experiment
kwargs = lbann.contrib.args.get_scheduler_kwargs(args)

lbann.contrib.launcher.run(trainer, model, data_reader, opt,
                           job_name=args.job_name,
                           **kwargs)
