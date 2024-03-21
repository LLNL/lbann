import lbann
import torch
import lbann.torch
import lbann.contrib.launcher
import lbann.contrib.args
import google.protobuf.text_format as txtf
import lbann.contrib.hyperparameter as hyper
import argparse
import sys
import os
from os.path import abspath, dirname, join

# ==============================================
# Setup 
# ==============================================

# Debugging
torch._dynamo.config.verbose=True

import models.probiesNet_HRRL_arch as model 

# Command-line arguments
desc = ('Reads in the HRRL model and runs it on HRRL PROBIES data. ')
parser = argparse.ArgumentParser(description=desc)
lbann.contrib.args.add_scheduler_arguments(parser)
parser.add_argument(
    '--job-name', action='store', default='probiesNet', type=str,
    help='scheduler job name (default: probiesNet)')
parser.add_argument(
    '--mini-batch-size', action='store', default=32, type=int,
    help='mini-batch size (default: 32)', metavar='NUM')
parser.add_argument(
    '--reader-prototext', action='store', default='probies_v2.prototext', type=str,
    help='data to use (default: probies_v2.prototext, 20K data)')
parser.add_argument(
    '--num-epochs', action='store', default=100, type=int, 
    help='number of epochs (default: 100)', metavar='NUM')

# Add reader prototext
lbann.contrib.args.add_optimizer_arguments(parser)
args = parser.parse_args()

# Default data reader
cur_dir = dirname(abspath(__file__))
data_reader_prototext = join(cur_dir,
                             'data',
                             args.reader_prototext)

print("DATA READER ", data_reader_prototext)

script = lbann.launcher.make_batch_script(nodes=2, procs_per_node=4) 

# Run experiment
kwargs = lbann.contrib.args.get_scheduler_kwargs(args)

images = lbann.Input(data_field='samples')
num_labels = 5
images = lbann.Reshape(images, dims=[1, 300, 300])
responses = lbann.Input(data_field='responses') #labels

# Initialize
mod = model.PROBIESNet()

graph = lbann.torch.compile(mod, x=torch.rand(64,1,300,300)) # batch size 64

images = graph[0]
pred = graph[-1] 

# Lbann slice layer 
pred_slice = lbann.Slice(pred, axis=0, slice_points=[0,1,2,3,4,5])
response_slice = lbann.Slice(responses, axis=0, slice_points=[0,1,2,3,4,5])

# ==============================================
# Metrics 
# ==============================================

# MSE loss between responses and preds 
mse = lbann.MeanSquaredError([responses, pred])

# Responses
epmax_response = lbann.Identity(response_slice)
etot_response = lbann.Identity(response_slice)
n_response = lbann.Identity(response_slice)
t_response = lbann.Identity(response_slice)
alpha_response = lbann.Identity(response_slice)

# Preds
epmax_pred = lbann.Identity(pred_slice)
etot_pred = lbann.Identity(pred_slice)
n_pred = lbann.Identity(pred_slice)
t_pred = lbann.Identity(pred_slice)
alpha_pred = lbann.Identity(pred_slice)

# MSEs
mse_epmax = lbann.MeanSquaredError([epmax_response, epmax_pred])
mse_etot = lbann.MeanSquaredError([etot_response, etot_pred])
mse_n = lbann.MeanSquaredError([n_response, n_pred])
mse_t = lbann.MeanSquaredError([t_response, t_pred])
mse_alpha = lbann.MeanSquaredError([alpha_response, alpha_pred])

layers = list(lbann.traverse_layer_graph([images, responses]))

# Append Metrics
metrics = [lbann.Metric(mse, name='mse')]
metrics.append(lbann.Metric(mse_epmax, name='mse_epmax'))
metrics.append(lbann.Metric(mse_etot, name='mse_etot'))
metrics.append(lbann.Metric(mse_n, name='mse_n'))
metrics.append(lbann.Metric(mse_t, name='mse_t'))
metrics.append(lbann.Metric(mse_alpha, name='mse_alpha'))

callbacks = [lbann.CallbackPrint(),
            lbann.CallbackTimer()]

layers = list(lbann.traverse_layer_graph([images, responses]))

model = lbann.Model(args.num_epochs,
                    layers=layers, 
                    objective_function=mse, 
                    metrics=metrics,
                    callbacks=callbacks) 

# Setup optimizer
opt = lbann.Adam(learn_rate=0.0002,beta1=0.9,beta2=0.99,eps=1e-8)

# Load data reader from prototext
data_reader_proto = lbann.lbann_pb2.LbannPB()
with open(data_reader_prototext, 'r') as f:
    txtf.Merge(f.read(), data_reader_proto)
data_reader_proto = data_reader_proto.data_reader

# Aliases for simplicity
SGD = lbann.BatchedIterativeOptimizer
RPE = lbann.RandomPairwiseExchange

# Construct the local training algorithm
local_sgd = SGD("local sgd", num_iterations=10)

# Construct the metalearning strategy
meta_learning = RPE(
    metric_strategies={'mse': RPE.MetricStrategy.LOWER_IS_BETTER})

# Setup vanilla trainer
trainer = lbann.Trainer(mini_batch_size=args.mini_batch_size)

lbann.contrib.launcher.run(trainer, model, data_reader_proto, opt, procs_per_trainer=1,
   lbann_args=" --use_data_store --preload_data_store", job_name=args.job_name, 
   binary_protobuf=True, **kwargs)
