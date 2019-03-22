#!/usr/bin/env python3
import argparse
import os.path
import google.protobuf.text_format as txtf
import lbann
from lbann.util import lbann_dir
import lbann.models
import lbann.proto
import lbann.contrib.args

# Command-line arguments
desc = ('Construct and run LeNet on MNIST data. '
        'Running the experiment is only supported on LC systems.')
parser = argparse.ArgumentParser(description=desc)
lbann.contrib.args.add_scheduler_arguments(parser)
parser.add_argument(
    '--prototext', action='store', type=str,
    help='exported prototext file', metavar='FILE')
parser.add_argument(
    '--disable-run', action='store_true',
    help='do not run experiment (e.g. if only the prototext is desired)')
args = parser.parse_args()

# Construct layer graph
input = lbann.Input()
images = lbann.Identity(input)
labels = lbann.Identity(input)
preds = lbann.models.LeNet(10)(images)
softmax = lbann.Softmax(preds)
loss = lbann.CrossEntropy([softmax, labels])
acc = lbann.CategoricalAccuracy([softmax, labels])

# Setup model
mini_batch_size = 64
num_epochs = 20
model = lbann.Model(mini_batch_size,
                    num_epochs,
                    layers=lbann.traverse_layer_graph(input),
                    objective_function=loss,
                    metrics=[lbann.Metric(acc, name='accuracy', unit='%')],
                    callbacks=[lbann.CallbackPrint(), lbann.CallbackTimer()])

# Setup optimizer
opt = lbann.SGD(learn_rate=0.01, momentum=0.9)

# Load data reader from prototext
data_reader_file = os.path.join(lbann_dir(),
                                'model_zoo',
                                'data_readers',
                                'data_reader_mnist.prototext')
data_reader_proto = lbann.lbann_pb2.LbannPB()
with open(data_reader_file, 'r') as f:
    txtf.Merge(f.read(), data_reader_proto)
data_reader_proto = data_reader_proto.data_reader

# Save prototext
if args.prototext:
    lbann.proto.save_prototext(args.prototext,
                               model=model, optimizer=opt,
                               data_reader=data_reader_proto)

# Run experiment
if not args.disable_run:
    from lbann.contrib.lc.launcher import run
    kwargs = {}
    if args.nodes:          kwargs['nodes'] = args.nodes
    if args.procs_per_node: kwargs['procs_per_node'] = args.procs_per_node
    if args.partition:      kwargs['partition'] = args.partition
    if args.account:        kwargs['account'] = args.account
    if args.time_limit:     kwargs['time_limit'] = args.time_limit
    run(model, data_reader_proto, opt,
        job_name = 'lbann_lenet',
        **kwargs)
