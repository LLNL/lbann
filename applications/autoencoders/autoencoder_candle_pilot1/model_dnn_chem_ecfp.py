################################################################################
# Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by the LBANN Research Team (B. Van Essen, et al.) listed in
# the CONTRIBUTORS file. <lbann-dev@llnl.gov>
#
# LLNL-CODE-697807.
# All rights reserved.
#
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
# FIXME: Is this a good description?
# model_dnn_chem_ecfp.py - A python run script for a CANcer
# Distributed Learning Environment (CANDLE) autoencoder example
#
################################################################################

import os.path
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
import pilot1


# Command-line arguments
desc = ('Construct and run CANDLE on Pilot1 data. '
        'Running the experiment is only supported on LC systems.')
parser = argparse.ArgumentParser(description=desc)
lbann.contrib.args.add_scheduler_arguments(parser)
parser.add_argument(
    '--job-name', action='store', default='lbann_CANDLE_pilot1', type=str,
    help='scheduler job name (default: lbann_CANDLE_pilot1)')
parser.add_argument(
    '--mini-batch-size', action='store', default=128, type=int,
    help='mini-batch size (default: 128)', metavar='NUM')
parser.add_argument(
    '--num-epochs', action='store', default=4, type=int,
    help='number of epochs (default: 4)', metavar='NUM')
parser.add_argument(
    '--data-reader', action='store', default='default', type=str,
    help='Data reader options: \"combo\", \"gdc\", \"growth\" or \"pilot1\" (default: data_reader_candle_pilot1.prototext)')
lbann.contrib.args.add_optimizer_arguments(parser, default_learning_rate=0.1)
args = parser.parse_args()


# Start of layers

# Construct layer graph
input_ = lbann.Input(name='data')
finetunedata = lbann.Split(input_, name='finetunedata')
label = lbann.Split(input_, name='label')

# Encoder
encode1 = lbann.FullyConnected(finetunedata,
                               name="encode1",
                               data_layout="model_parallel",
                               num_neurons=2000,
                               has_bias=True)

relu1 = lbann.Relu(encode1, name="relu1", data_layout="model_parallel")

encode2 = lbann.FullyConnected(relu1,
                               name="encode2",
                               data_layout="model_parallel",
                               num_neurons=1000,
                               has_bias=True)

relu2 = lbann.Relu(encode2, name="relu2", data_layout="model_parallel")

encode3 = lbann.FullyConnected(relu2,
                               name="encode3",
                               data_layout="model_parallel",
                               num_neurons=500,
                               has_bias=True)

relu3 = lbann.Relu(encode3, name="relu3", data_layout="model_parallel")

encode4 = lbann.FullyConnected(relu3,
                               name="encode4",
                               data_layout="model_parallel",
                               num_neurons=250,
                               has_bias=True)

relu4 = lbann.Relu(encode4, name="relu4", data_layout="model_parallel")

encode5 = lbann.FullyConnected(relu4,
                               name="encode5",
                               data_layout="model_parallel",
                               num_neurons=100,
                               has_bias=True)

relu5 = lbann.Relu(encode5, name="relu5", data_layout="model_parallel")

ip2 = lbann.FullyConnected(relu5,
                           name="ip2",
                           data_layout="model_parallel",
                           num_neurons=2,
                           has_bias=True)

prob = lbann.Softmax(ip2, name="prob", data_layout="model_parallel")

cross_entropy = lbann.CrossEntropy([prob, label],
                                   name="cross_entropy",
                                   data_layout="model_parallel")

categorical_accuracy = lbann.CategoricalAccuracy([prob, label],
                                                 name="categorical_accuracy",
                                                 data_layout="model_parallel")

layer_list = list(lbann.traverse_layer_graph(input_))

# Set up objective function
layer_term = lbann.LayerTerm(cross_entropy)
obj = lbann.ObjectiveFunction(layer_term)

# Metrics
metrics = [lbann.Metric(categorical_accuracy, name="accuracy")]

# Callbacks
callbacks = [lbann.CallbackPrint(),
             lbann.CallbackTimer()]

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
data_reader_prefix = 'data_reader_candle_pilot1'
if args.data_reader == "default" or args.data_reader == "pilot1":
  data_reader_file = data_reader_prefix + '.prototext'
elif args.data_reader == "combo":
  data_reader_file = data_reader_prefix + '_combo.prototext'
elif args.data_reader == "gdc":
  data_reader_file = data_reader_prefix + '_gdc.prototext'
elif args.data_reader == "growth":
  data_reader_file = data_reader_prefix + '_growth.prototext'
else:
  raise InvalidOption('Data reader selection \"' + args.data_reader + '\" is invalid. Use \"combo\", \"gdc\", or \"growth\". Default is data_reader_candle_pilot1.prototext.')

data_reader = pilot1.make_data_reader(data_reader_file)

# Setup trainer
trainer = lbann.Trainer(mini_batch_size=args.mini_batch_size)

# Run experiment
kwargs = lbann.contrib.args.get_scheduler_kwargs(args)

lbann.contrib.launcher.run(trainer, model, data_reader, opt,
                           job_name=args.job_name,
                           **kwargs)
