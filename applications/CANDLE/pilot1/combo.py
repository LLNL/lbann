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
# combo.py - A python run script for a CANcer Distributed
# Learning Environment (CANDLE) autoencoder example
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
    '--mini-batch-size', action='store', default=256, type=int,
    help='mini-batch size (default: 256)', metavar='NUM')
parser.add_argument(
    '--num-epochs', action='store', default=10, type=int,
    help='number of epochs (default: 10)', metavar='NUM')
parser.add_argument(
    '--random-seed', action='store', default=0, type=int,
    help='random seed for LBANN RNGs', metavar='NUM')
parser.add_argument(
    '--data-reader', action='store', default='default', type=str,
    help='Data reader options: \"combo\", \"gdc\", \"growth\", or \"pilot1\" (default: data_reader_candle_pilot1.prototext)')
lbann.contrib.args.add_optimizer_arguments(parser, default_learning_rate=0.1)
args = parser.parse_args()


# Start of layers

# Construct layer graph
input_ = lbann.Input(name='input', target_mode="regression")
data = lbann.Split(input_, name='data')
response = lbann.Split(input_, name="response")

# SLICE
slice_data = lbann.Slice(data,
                         name="slice_data",
                         slice_points=[0, 921, 4750, 8579],
                         data_layout="model_parallel")

# Gene Track
gene_fc1 = lbann.FullyConnected(slice_data,
                                name="gene_fc1",
                                data_layout="model_parallel",
                                num_neurons=1000,
                                has_bias=True)

# Relu
gene_relu1 = lbann.Relu(gene_fc1,
                        name="gene_relu1",
                        data_layout="model_parallel")

# Dropout
gene_dropout1 = lbann.Dropout(gene_relu1,
                              name="gene_dropout1",
                              data_layout="model_parallel",
                              keep_prob=0.95)

# Fully Connected
gene_fc2 = lbann.FullyConnected(gene_dropout1,
                                name="gene_fc2",
                                data_layout="model_parallel",
                                num_neurons=1000,
                                has_bias=True)

# Relu
gene_relu2 = lbann.Relu(gene_fc2,
                        name="gene_relu2",
                        data_layout="model_parallel")

# Dropout
gene_dropout2 = lbann.Dropout(gene_relu2,
                              name="gene_dropout2",
                              keep_prob=0.95)

# Fully Connected
gene_fc3 = lbann.FullyConnected(gene_dropout2,
                                name="gene_fc3",
                                data_layout="model_parallel",
                                num_neurons=1000,
                                has_bias=True)

# Relu
gene_relu3 = lbann.Relu(gene_fc3,
                        name="gene_relu3",
                        data_layout="model_parallel")

# Dropout
gene_dropout3 = lbann.Dropout(gene_relu3,
                              name="gene_dropout3",
                              data_layout="model_parallel",
                              keep_prob=0.95)

# Shared Weights for Drug Tracks
drug_fc1_w = lbann.Initializer()
drug_fc2_w = lbann.Initializer()
drug_fc3_w = lbann.Initializer()

# Drug1 Track
drug1_fc1 = lbann.FullyConnected(slice_data,
                                 name="drug1_fc1",
                                 data_layout="model_parallel",
                                 weights=drug_fc1_w,
                                 num_neurons=1000,
                                 has_bias=True)

# Relu
drug1_relu1 = lbann.Relu(drug1_fc1,
                         name="drug1_relu1",
                         data_layout="model_parallel")

# Dropout
drug1_dropout1 = lbann.Dropout(drug1_relu1,
                               name="drug1_dropout1",
                               data_layout="model_parallel",
                               keep_prob=0.95)

# Fully Connected
drug1_fc2 = lbann.FullyConnected(drug1_dropout1,
                                 name="drug1_fc2",
                                 data_layout="model_parallel",
                                 weights=drug_fc2_w,
                                 num_neurons=1000,
                                 has_bias=True)

# Relu
drug1_relu2 = lbann.Relu(drug1_fc2,
                         name="drug1_relu2",
                         data_layout="model_parallel")

# Dropout
drug1_dropout2 = lbann.Dropout(drug1_relu2,
                               name="drug1_dropout2",
                               data_layout="model_parallel",
                               keep_prob=0.95)

# Fully Connected
drug1_fc3 = lbann.FullyConnected([drug1_dropout2],
                                 name="drug_fc3",
                                 data_layout="model_parallel",
                                 weights=drug_fc3_w,
                                 num_neurons=1000,
                                 has_bias=True)
# Relu
drug1_relu3 = lbann.Relu(drug1_fc3,
                         name="drug1_relu3",
                         data_layout="model_parallel")

# Dropout
drug1_dropout3 = lbann.Dropout(drug1_relu3,
                               name="drug1_dropout3",
                               data_layout="model_parallel",
                               keep_prob=0.95)

# Drug2 Track

drug2_fc1 = lbann.FullyConnected(slice_data,
                                 name="drug2_fc1",
                                 data_layout="model_parallel",
                                 weights=drug_fc1_w,
                                 num_neurons=1000,
                                 has_bias=True)

# Relu
drug2_relu1 = lbann.Relu(drug2_fc1,
                         name="drug2_relu1",
                         data_layout="model_parallel")

# Dropout
drug2_dropout1 = lbann.Dropout(drug2_relu1,
                               name="drug2_dropout1",
                               data_layout="model_parallel",
                               keep_prob=0.95)

# Fully Connected
drug2_fc2 = lbann.FullyConnected(drug2_dropout1,
                                 name="drug2_fc2",
                                 data_layout="model_parallel",
                                 weights=drug_fc2_w,
                                 num_neurons=1000,
                                 has_bias=True)

# Relu
drug2_relu2 = lbann.Relu(drug1_fc2,
                         name="drug2_relu2",
                         data_layout="model_parallel")

# Dropout
drug2_dropout2 = lbann.Dropout(drug2_relu2,
                               name="drug2_dropout2",
                               data_layout="model_parallel",
                               keep_prob=0.95)

# Fully Connected
drug2_fc3 = lbann.FullyConnected(drug2_dropout2,
                                 name="drug2_fc3",
                                 data_layout="model_parallel",
                                 weights=drug_fc3_w,
                                 num_neurons=1000,
                                 has_bias=True)

# Relu
drug2_relu3 = lbann.Relu(drug2_fc3,
                         name="drug2_relu3",
                         data_layout="model_parallel")

# Dropout
drug2_dropout3 = lbann.Dropout(drug2_relu3,
                               name="drug2_dropout3",
                               data_layout="model_parallel",
                               keep_prob=0.95)

# Concat
concat = lbann.Concatenation([gene_dropout3, drug1_dropout3, drug2_dropout3],
                             name="concat",
                             data_layout="model_parallel")

# Combined Track
combined_fc1 = lbann.FullyConnected(concat,
                                    name="combined_fc1",
                                    data_layout="model_parallel",
                                    num_neurons=1000,
                                    has_bias=True)

# Relu
combined_relu1 = lbann.Relu(combined_fc1,
                            name="combined_relu1",
                            data_layout="model_parallel")

# Dropout
combined_dropout1 = lbann.Dropout(combined_relu1,
                                  name="combined_dropout1",
                                  data_layout="model_parallel",
                                  keep_prob=0.95)

# Fully Connected
combined_fc2 = lbann.FullyConnected(combined_dropout1,
                                    name="combined_fc2",
                                    data_layout="model_parallel",
                                    num_neurons=1000,
                                    has_bias=True)

# Relu
combined_relu2 = lbann.Relu(combined_fc2,
                            name="combined_relu2",
                            data_layout="model_parallel")

# Dropout
combined_dropout2 = lbann.Dropout(combined_relu2,
                                  name="combined_dropout2",
                                  data_layout="model_parallel",
                                  keep_prob=0.95)

# Fully Connected
combined_fc3 = lbann.FullyConnected(combined_dropout2,
                                    name="combined_fc3",
                                    data_layout="model_parallel",
                                    num_neurons=1000,
                                    has_bias=True)

# Relu
combined_relu3 = lbann.Relu(combined_fc3,
                            name="combined_relu3",
                            data_layout="model_parallel")

# Dropout
combined_dropout3 = lbann.Dropout(combined_relu3,
                                  name="combined_dropout3",
                                  data_layout="model_parallel",
                                  keep_prob=0.95)

# Fully Connected
fc = lbann.FullyConnected(combined_dropout3,
                          name="fc",
                          data_layout="model_parallel",
                          num_neurons=1,
                          has_bias=True)

mse = lbann.MeanSquaredError([fc, response],
                             name="mse",
                             data_layout="model_parallel")

r2_var = lbann.Variance(fc,
                        name="r2_var",
                        data_layout="model_parallel",
                        biased=True)

r2_div = lbann.Divide([mse, r2_var],
                       name="r2_div",
                       data_layout="model_parallel")

r2_one = lbann.Constant(name="model_parallel",
                       value=1,
                       num_neurons=1)

r2 = lbann.Subtract([r2_one, r2_div],
                    name="r2",
                    data_layout="model_parallel")

layer_list = list(lbann.traverse_layer_graph(input_))

# Set up objective function
layer_term = lbann.LayerTerm(mse)
obj = lbann.ObjectiveFunction(layer_term)

# Metrics
metrics = [lbann.Metric(mse, name="mean squared error"),
           lbann.Metric(r2, name="R2")]

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
if args.data_reader == "default":
  data_reader_file = data_reader_prefix + '.prototext'
elif args.data_reader == "combo":
  data_reader_file = data_reader_prefix + '_combo.prototext'
elif args.data_reader == "gdc":
  data_reader_file = data_reader_prefix + '_gdc.prototext'
elif args.data_reader == "growth":
  data_reader_file = data_reader_prefix + '_growth.prototext'
else:
  raise InvalidOption('Data reader selection \"' + args.data_reader + '\" is invalid. Use \"combo\", \"gdc\", \"growth\", or \"pilot1\". Default is data_reader_candle_pilot1.prototext.')

data_reader = pilot1.make_data_reader(data_reader_file)

# Setup trainer
trainer = lbann.Trainer(mini_batch_size=args.mini_batch_size)

# Run experiment
kwargs = lbann.contrib.args.get_scheduler_kwargs(args)

lbann.contrib.launcher.run(trainer, model, data_reader, opt,
                           job_name=args.job_name,
                           **kwargs)
