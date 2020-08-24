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
# ae_nodeselect_gdc.py - A python run script for a CANcer Distributed
# Learning Environment (CANDLE) autoencoder example
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


# Command-line arguments
desc = ('Construct and run CANDLE on Pilot1 data. '
        'Running the experiment is only supported on LC systems.')
parser = argparse.ArgumentParser(description=desc)
lbann.contrib.args.add_scheduler_arguments(parser)
parser.add_argument(
    '--job-name', action='store', default='lbann_image_ae', type=str,
    help='scheduler job name (default: lbann_CANDLE)')
parser.add_argument(
    '--mini-batch-size', action='store', default=128, type=int,
    help='mini-batch size (default: 50)', metavar='NUM')
parser.add_argument(
    '--num-epochs', action='store', default=90, type=int,
    help='number of epochs (default: 20)', metavar='NUM')
parser.add_argument(
    '--num-classes', action='store', default=1000, type=int,
    help='number of ImageNet classes (default: 1000)', metavar='NUM')
parser.add_argument(
    '--random-seed', action='store', default=0, type=int,
    help='random seed for LBANN RNGs', metavar='NUM')
parser.add_argument(
    '--data-reader', action='store',
    default='../../../data_readers/data_reader_candle_pilot1_gdc.prototext', type=str,
    help='scheduler job name (default: data_readers/data_reader_candle_pilot1_gdc.prototext)')
lbann.contrib.args.add_optimizer_arguments(parser, default_learning_rate=0.1)
args = parser.parse_args()


# Start of layers

# Construct layer graph
input_ = lbann.Input(name='data', target_mode="N/A")
recon_data = lbann.Identity(input_, name='recon_data')

# Encoder

w1 = lbann.Initializer()

encode1 = lbann.FullyConnected(recon_data,
                               name="encode1",
                               data_layout="model_parallel",
                               weights="w1",
                               num_neurons=1000,
                               has_bias=True,
                               transpose=False)

sigmoid1 = lbann.Sigmoid(encode1,
                         name="sigmoid1",
                         data_layout="model_parallel")

decode1 = lbann.FullyConnected(sigmoid1,
                               name="decode1",
                               data_layout="model_parallel",
                               weights=w1,
                               #num_neurons_from_data_reader=True,
                               has_bias=True,
                               transpose=True)

sigmoid2 = lbann.Sigmoid(decode1,
                         name="sigmoid2",
                         data_layout="model_parallel")

# pearson_r: A Tensor representing the current Pearson product-moment
# correlation coefficient, the value of cov(predictions, labels) /
# sqrt(var(predictions) * var(labels))

unbiased_covariance = lbann.Covariance([sigmoid2, recon_data],
                                       name="unbiased_covariance",
                                       biased=False,
                                       data_layout="model_parallel")

pred_variance = lbann.Variance(sigmoid2,
                               name="pred_variance",
                               biased=False,
                               data_layout="model_parallel")

data_variance = lbann.Variance(recon_data,
                               name="data_variance",
                               biased=False,
                               data_layout="model_parallel")

mult = lbann.Multiply([pred_variance, data_variance],
                      name="mult",
                      data_layout="model_parallel")

sqrt = lbann.Sqrt(mult, name="sqrt", data_layout="model_parallel")

pearson_r = lbann.Divide([unbiased_covariance, sqrt],
                         name="pearson_r",
                         data_layout="model_parallel")

mse = lbann.MeanSquaredError([recon_data, sigmoid2],
                             data_layout="model_parallel")

layer_list = list(lbann.traverse_layer_graph(input_))

# Set up objective function
layer_term = lbann.LayerTerm(mse)
obj = lbann.ObjectiveFunction(layer_term)

# Metrics
metrics = [lbann.Metric(pearson_r, name="pearson_r")]

# Callbacks
callbacks = [lbann.CallbackPrint()]

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
data_reader_file = args.data_reader
data_reader_proto = lbann.lbann_pb2.LbannPB()
with open(data_reader_file, 'r') as f:
    txtf.Merge(f.read(), data_reader_proto)
data_reader_proto = data_reader_proto.data_reader

# Setup trainer
trainer = lbann.Trainer(mini_batch_size=args.mini_batch_size)

# Run experiment
kwargs = lbann.contrib.args.get_scheduler_kwargs(args)

lbann.contrib.launcher.run(trainer, model, data_reader_proto, opt,
                           job_name=args.job_name,
                           **kwargs)
