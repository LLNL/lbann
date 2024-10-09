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

# Command-line arguments
desc = ('Construct and a grid search on HRRL PROBIES data. ')
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

# Make script with 2 nodes
script = lbann.launcher.make_batch_script(nodes=2, procs_per_node=4) 

# Run experiment
kwargs = lbann.contrib.args.get_scheduler_kwargs(args)

# Define the model for the search
def make_search_model(
        learning_rate,
    	beta1, 
        beta2,
        eps,
        intermed_fc_layers,
        activation,
        dropout_percent,
        num_labels=5):
    
    import models.probiesNetLBANN_grid_search as model
    images = lbann.Input(data_field='samples')
    responses = lbann.Input(data_field='responses')
    num_labels = 5
    images = lbann.Reshape(images, dims=[1, 300, 300])
    pred = model.PROBIESNetLBANN(num_labels, intermed_fc_layers, activation, dropout_percent)(images)

    # lbann slice layer 
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

    # Append metrics
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
    opt = lbann.Adam(learn_rate=learning_rate,beta1=beta1,beta2=beta2,eps=eps)

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

    # Construct the metalearning strategy. 
    meta_learning = RPE(
        metric_strategies={'mse': RPE.MetricStrategy.LOWER_IS_BETTER})

    # Setup vanilla trainer
    trainer = lbann.Trainer(mini_batch_size=args.mini_batch_size)
    return model, opt, data_reader_proto, trainer

# run the grid search using make_search_model.  Options below.
hyper.grid_search(
    script,
    make_search_model,
    use_data_store=True, # must be True for images to scalers training
    procs_per_trainer=1,
    learning_rate=[0.00001],
    beta1=[0.9],
    beta2=[0.9],
    eps=[1e-8],
    intermed_fc_layers = [[960,240]],
    activation = [lbann.Relu],
    dropout_percent = [0.7])

# Sample syntax for a larger search:
# hyper.grid_search(
#     script,
#     make_search_model,
#     use_data_store=False,
#     procs_per_trainer=1,
#     learning_rate=[0.00001],
#     beta1=[0.9,0.99],
#     beta2=[0.9,0.99],
#     eps=[1e-8],
#     intermed_fc_layers = [[960,240],[1920,960,480,240],[480,240]],
#     activation = [lbann.Relu,lbann.Softmax,lbann.LeakyRelu],
#     dropout_percent = [0.3, 0.5, 0.7])
