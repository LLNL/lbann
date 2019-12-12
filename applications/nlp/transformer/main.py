import argparse
import math
import os.path
import sys

import lbann
import lbann.contrib.lc.launcher
import lbann.contrib.args
from lbann.util import str_list

# Local imports
current_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)
import dataset
import model

# ----------------------------------
# Options
# ----------------------------------

# Command-line arguments
parser = argparse.ArgumentParser()
lbann.contrib.args.add_scheduler_arguments(parser)
parser.add_argument(
    '--job-name', action='store', default='lbann_transformer', type=str,
    help='job name', metavar='NAME')
parser.add_argument(
    '--mini-batch-size', action='store', default=256, type=int,
    help='mini-batch size (default: 256)', metavar='NUM')
parser.add_argument(
    '--num-epochs', action='store', default=20, type=int,
    help='number of epochs (default: 20)', metavar='NUM')
parser.add_argument(
    '--num-attention-heads', action='store', default=8, type=int,
    help='number of parallel attention layers (default: 8)', metavar='NUM')
parser.add_argument(
    '--embed-dim', action='store', default=512, type=int,
    help='embedding space dimensions (default: 512)', metavar='NUM')
args = parser.parse_args()

# ----------------------------------
# Construct layer graph
# ----------------------------------

# Dataset properties
vocab_size = dataset.corpus.vocab_size
sequence_length = dataset.sample_dims()[0]

# Split input into sequence of token IDs
# Note: Sequence of token IDs and sequence of embeddings
input_ = lbann.Identity(lbann.Input())
input_slice = lbann.Slice(
    input_,
    slice_points=str_list(range(sequence_length+1)),
)
tokens = [lbann.Identity(input_slice) for _ in range(sequence_length)]

# Get sequence of embedding vectors
embedding_weights = lbann.Weights(
    initializer=lbann.NormalInitializer(mean=0, standard_deviation=1),
    name='embeddings',
)
embeddings = []
for token in tokens:
    embedding = lbann.Embedding(
        token,
        num_embeddings=vocab_size,
        embedding_dim=args.embed_dim,
        weights=embedding_weights,
    )
    embedding = lbann.WeightedSum(
        embedding,
        scaling_factors=str(math.sqrt(args.embed_dim)),
    )
    embeddings.append(embedding)

# Apply transformer model
model = model.Transformer(
    hidden_size=args.embed_dim,
    num_heads=args.num_attention_heads,
)
result = model(embeddings, embeddings[:-1])

# Predict next token in sequence
pred_fc = lbann.modules.FullyConnectedModule(
    vocab_size,
    transpose=True,
    activation=lbann.Softmax,
)
preds = [pred_fc(x) for x in result]

# Cross entropy loss
loss = []
acc = []
for i in range(sequence_length-1):
    pred = preds[i]
    label = lbann.OneHot(tokens[i+1], size=vocab_size)
    loss.append(lbann.CrossEntropy(pred, label))
    acc.append(lbann.TopKCategoricalAccuracy(pred, label, k=10))
acc = lbann.Reduction(lbann.Concatenation(acc), mode='average')

# ----------------------------------
# Create data reader
# ----------------------------------

reader = lbann.reader_pb2.DataReader()
_reader = reader.reader.add()
_reader.name = 'python'
_reader.role = 'train'
_reader.percent_of_data_to_use = 1.0
_reader.python.module = 'dataset'
_reader.python.module_dir = current_dir
_reader.python.sample_function = 'get_sample'
_reader.python.num_samples_function = 'num_samples'
_reader.python.sample_dims_function = 'sample_dims'

# ----------------------------------
# Run LBANN
# ----------------------------------

# Create LBANN objects
trainer = lbann.Trainer()
metrics = [lbann.Metric(acc, name='top-10 accuracy', unit='%')]
callbacks = [lbann.CallbackPrint(),
             lbann.CallbackTimer()]
model = lbann.Model(args.mini_batch_size,
                    args.num_epochs,
                    layers=lbann.traverse_layer_graph(input_),
                    objective_function=loss,
                    metrics=metrics,
                    callbacks=callbacks)
opt = lbann.SGD(learn_rate=0.01, momentum=0.9) # TODO: Adam with LR schedule

# Run LBANN
kwargs = lbann.contrib.args.get_scheduler_kwargs(args)
lbann.contrib.lc.launcher.run(trainer, model, reader, opt,
                              job_name=args.job_name,
                              **kwargs)
