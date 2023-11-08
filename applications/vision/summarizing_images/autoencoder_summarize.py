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
# autoencoder_summarize.py - A simple autoencoder for image data
# (Supports CIFAR-10 or Imagenet 1K)
#
# This example demonstrates the use of the image summarizer in
# autoencoder mode.
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
import cifar10
import imagenet

# Command-line arguments
desc = ('Construct and run ResNet on ImageNet-1K data. '
        'Running the experiment is only supported on LC systems.')
parser = argparse.ArgumentParser(description=desc)
lbann.contrib.args.add_scheduler_arguments(parser, 'lbann_image_ae')
parser.add_argument(
    '--width', action='store', default=1, type=float,
    help='Wide ResNet width factor (default: 1)')
parser.add_argument(
    '--bn-statistics-group-size', action='store', default=1, type=int,
    help=('Group size for aggregating batch normalization statistics '
          '(default: 1)'))
parser.add_argument(
    '--warmup', action='store_true', help='use a linear warmup')
parser.add_argument(
    '--mini-batch-size', action='store', default=256, type=int,
    help='mini-batch size (default: 256)', metavar='NUM')
parser.add_argument(
    '--num-epochs', action='store', default=90, type=int,
    help='number of epochs (default: 90)', metavar='NUM')
parser.add_argument(
    '--num-classes', action='store', default=1000, type=int,
    help='number of ImageNet classes (default: 1000)', metavar='NUM')
parser.add_argument(
    '--random-seed', action='store', default=0, type=int,
    help='random seed for LBANN RNGs', metavar='NUM')
parser.add_argument(
    '--dataset', action='store', default='imagenet', type=str,
    help='dataset to use; \"cifar10\" or \"imagenet\"')
parser.add_argument(
    '--data-reader-fraction', action='store',
    default=1.0, type=float,
    help='the fraction of the data to use (default: 1.0)', metavar='NUM')
lbann.contrib.args.add_optimizer_arguments(parser, default_learning_rate=0.1)
args = parser.parse_args()

# Due to a data reader limitation, the actual model realization must be
# hardcoded to 1000 labels for ImageNet; 10 for CIFAR10.
dataset = args.dataset;
if dataset == 'imagenet':
    num_labels=1000
elif dataset == 'cifar10':
    num_labels=10
else:
    print("Dataset must be cifar10 or imagenet. Try again.")
    exit()

# Construct layer graph
input_ = lbann.Input(name='input', target_mode='classification')
image = lbann.Identity(input_, name='images')
dummy = lbann.Dummy(input_, name='labels')

# Encoder
encode1 = lbann.FullyConnected(image,
                               name="encode1",
                               data_layout="model_parallel",
                               num_neurons=1000,
                               has_bias=True)

relu1 = lbann.Relu(encode1, name="relu1", data_layout="model_parallel")

dropout1 = lbann.Dropout(relu1,
                         name="dropout1",
                         data_layout="model_parallel",
                         keep_prob=0.8)

decode1 = lbann.FullyConnected(dropout1,
                               name="decode1",
                               data_layout="model_parallel",
                               hint_layer=image,
                               has_bias=True)

reconstruction = lbann.Sigmoid(decode1,
                               name="reconstruction",
                               data_layout="model_parallel")

dropout2 = lbann.Dropout(reconstruction,
                         name="dropout2",
                         data_layout="model_parallel",
                         keep_prob=0.8)


# Reconstruction
mean_squared_error = lbann.MeanSquaredError([dropout2, image],
                             name="mean_squared_error")

layer_term = lbann.LayerTerm(mean_squared_error)
obj = lbann.ObjectiveFunction(layer_term)

metrics = [lbann.Metric(mean_squared_error, name=mean_squared_error.name)]

img_strategy = lbann.TrackSampleIDsStrategy(
    input_layer_name=input_.name,
    num_tracked_images=10)

summarize_images = lbann.CallbackSummarizeImages(
    selection_strategy=img_strategy,
    image_source_layer_name=reconstruction.name,
    epoch_interval=1)

# Dump original image from input layer one time
summarize_input_layer = lbann.CallbackSummarizeImages(
    selection_strategy=img_strategy,
    image_source_layer_name=input_.name,
    epoch_interval=10000)

callbacks = [lbann.CallbackPrint(),
             lbann.CallbackTimer(),
             summarize_input_layer,
             summarize_images]

layer_list = list(lbann.traverse_layer_graph(input_))
model = lbann.Model(args.num_epochs,
                    layers=layer_list,
                    objective_function=obj,
                    metrics=metrics,
                    callbacks=callbacks,
                    summary_dir=".")

# Setup optimizer
opt = lbann.contrib.args.create_optimizer(args)

# Setup data reader
num_classes=min(args.num_classes, num_labels)

if dataset == "cifar10":
    data_reader = cifar10.make_data_reader(num_classes=num_classes)
else:
    data_reader = imagenet.make_data_reader(num_classes=num_classes)

# Setup trainer
trainer = lbann.Trainer(random_seed=args.random_seed, mini_batch_size=args.mini_batch_size)

# Run experiment
kwargs = lbann.contrib.args.get_scheduler_kwargs(args)
kwargs['lbann_args'] = '--data_reader_fraction='+str(args.data_reader_fraction)

lbann.contrib.launcher.run(trainer, model, data_reader, opt,
                           job_name=args.job_name,
                           **kwargs)
