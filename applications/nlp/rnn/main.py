"""Simple recurrent network on tokenized text data."""
import argparse
import os.path
import sys

import lbann
import lbann.modules
import lbann.contrib.launcher
import lbann.contrib.args

# Local imports
current_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)
import dataset

# ----------------------------------
# Options
# ----------------------------------

# Command-line arguments
parser = argparse.ArgumentParser()
lbann.contrib.args.add_scheduler_arguments(parser, 'lbann_textrnn')
parser.add_argument(
    '--mini-batch-size', action='store', default=256, type=int,
    help='mini-batch size (default: 256)', metavar='NUM')
parser.add_argument(
    '--num-epochs', action='store', default=20, type=int,
    help='number of epochs (default: 20)', metavar='NUM')
parser.add_argument(
    '--latent-dim', action='store', default=128, type=int,
    help='latent space dimensions (default: 128)', metavar='NUM')
args = parser.parse_args()

# ----------------------------------
# Construct layer graph
# ----------------------------------

# Dataset properties
vocab_size = dataset.corpus.vocab_size
sequence_length = dataset.sample_dims()[0]

# Input is a sequence of token IDs
input_ = lbann.Input(data_field='samples')
input_slice = lbann.Slice(input_,
                          slice_points=range(sequence_length+1))
tokens_list = [lbann.Identity(input_slice) for _ in range(sequence_length)]

# Get sequence of embedding vectors
embeddings = lbann.Embedding(input_,
                             num_embeddings=vocab_size,
                             embedding_dim=args.latent_dim)
embeddings_slice = lbann.Slice(embeddings,
                               axis=0,
                               slice_points=range(sequence_length+1))
embeddings_list = [lbann.Reshape(embeddings_slice, dims=[-1])
                   for _ in range(sequence_length)]

# Layer modules
lstm = lbann.modules.LSTMCell(args.latent_dim)
lstm_state = [lbann.Constant(value=0, num_neurons=args.latent_dim),
              lbann.Constant(value=0, num_neurons=args.latent_dim)]
pred_fc = lbann.modules.FullyConnectedModule(vocab_size,
                                             data_layout='model_parallel')

# Iterate through RNN steps
loss = []
for step in range(sequence_length-1):

    # Predict next token with RNN
    x = embeddings_list[step]
    x, lstm_state = lstm(x, lstm_state)
    x = pred_fc(x)
    pred = lbann.Softmax(x)

    # Evaluate prediction with cross entropy
    ground_truth = lbann.OneHot(tokens_list[step+1], size=vocab_size)
    cross_entropy = lbann.CrossEntropy([pred, ground_truth])
    loss.append(lbann.LayerTerm(cross_entropy, scale=1/(sequence_length-1)))

# ----------------------------------
# Create data reader
# ----------------------------------

reader = lbann.reader_pb2.DataReader()
_reader = reader.reader.add()
_reader.name = 'python'
_reader.role = 'train'
_reader.shuffle = True
_reader.fraction_of_data_to_use = 1.0
_reader.python.module = 'dataset'
_reader.python.module_dir = current_dir
_reader.python.sample_function = 'get_sample'
_reader.python.num_samples_function = 'num_samples'
_reader.python.sample_dims_function = 'sample_dims'

# ----------------------------------
# Run LBANN
# ----------------------------------

# Create LBANN objects
trainer = lbann.Trainer(mini_batch_size=args.mini_batch_size)
model = lbann.Model(args.num_epochs,
                    layers=lbann.traverse_layer_graph(input_),
                    objective_function=loss,
                    callbacks=[lbann.CallbackPrint(),
                               lbann.CallbackTimer()])
opt = lbann.SGD(learn_rate=0.01, momentum=0.9)

# Run LBANN
kwargs = lbann.contrib.args.get_scheduler_kwargs(args)
lbann.contrib.launcher.run(trainer, model, reader, opt,
                           job_name=args.job_name,
                           **kwargs)
