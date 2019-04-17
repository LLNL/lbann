#!/usr/bin/env python3
import argparse
from os.path import join
import google.protobuf.text_format as txtf
import lbann
import lbann.models
import lbann.proto
import lbann.contrib.args

# Command-line arguments
desc = ('Construct and run AlexNet on ImageNet-1K data. '
        'Running the experiment is only supported on LC systems.')
data_reader_prototext = join(lbann.lbann_dir(),
                             'model_zoo',
                             'data_readers',
                             'data_reader_imagenet.prototext')
parser = argparse.ArgumentParser(description=desc)
lbann.contrib.args.add_scheduler_arguments(parser)
parser.add_argument(
    '--mini-batch-size', action='store', default=256, type=int,
    help='mini-batch size (default: 256)', metavar='NUM')
parser.add_argument(
    '--num-epochs', action='store', default=100, type=int,
    help='number of epochs (default: 100)', metavar='NUM')
parser.add_argument(
    '--num-labels', action='store', default=1000, type=int,
    help='number of data classes (default: 1000)', metavar='NUM')
lbann.contrib.args.add_optimizer_arguments(parser)
parser.add_argument(
    '--data-reader', action='store',
    default=data_reader_prototext, type=str,
    help='data reader prototext file (default: ' + data_reader_prototext + ')',
    metavar='FILE')
parser.add_argument(
    '--imagenet-classes', action='store', type=int,
    help='number of ImageNet-1K classes (availability of subsampled datasets may vary by system)',
    metavar='NUM')
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
preds = lbann.models.AlexNet(args.num_labels)(images)
probs = lbann.Softmax(preds)
cross_entropy = lbann.CrossEntropy([probs, labels])
top1 = lbann.CategoricalAccuracy([probs, labels])
top5 = lbann.TopKCategoricalAccuracy([probs, labels], k=5)
layers = list(lbann.traverse_layer_graph(input))

# Setup objective function
weights = set()
for l in layers:
    weights.update(l.weights)
l2_reg = lbann.L2WeightRegularization(weights=weights, scale=5e-4)
obj = lbann.ObjectiveFunction([cross_entropy, l2_reg])

# Setup model
metrics = [lbann.Metric(top1, name='top-1 accuracy', unit='%'),
           lbann.Metric(top5, name='top-5 accuracy', unit='%')]
callbacks = [lbann.CallbackPrint(),
             lbann.CallbackTimer(),
             lbann.CallbackDropFixedLearningRate(
                 drop_epoch=[20,40,60], amt=0.1)]
model = lbann.Model(args.mini_batch_size,
                    args.num_epochs,
                    layers=layers,
                    weights=weights,
                    objective_function=obj,
                    metrics=metrics,
                    callbacks=callbacks)

# Setup optimizer
opt = lbann.contrib.args.create_optimizer(args)

# Load data reader from prototext
data_reader_proto = lbann.lbann_pb2.LbannPB()
with open(args.data_reader, 'r') as f:
  txtf.Merge(f.read(), data_reader_proto)
data_reader_proto = data_reader_proto.data_reader

# Setup trainer
trainer = lbann.Trainer()

# Save prototext
if args.prototext:
    lbann.proto.save_prototext(args.prototext,
                               trainer=trainer,
                               model=model, optimizer=opt,
                               data_reader=data_reader_proto)

# Run experiment
if not args.disable_run:
    from lbann.contrib.lc.paths import imagenet_dir, imagenet_labels
    import lbann.contrib.lc.launcher
    kwargs = {}
    if args.nodes:          kwargs['nodes'] = args.nodes
    if args.procs_per_node: kwargs['procs_per_node'] = args.procs_per_node
    if args.partition:      kwargs['partition'] = args.partition
    if args.account:        kwargs['account'] = args.account
    if args.time_limit:     kwargs['time_limit'] = args.time_limit
    if args.imagenet_classes:
        classes = args.imagenet_classes
        kwargs['lbann_args'] = (
            '--data_filedir_train={} --data_filename_train={} '
            '--data_filedir_test={} --data_filename_test={}'
            .format(imagenet_dir(data_set='train', num_classes=classes),
                    imagenet_labels(data_set='train', num_classes=classes),
                    imagenet_dir(data_set='val', num_classes=classes),
                    imagenet_labels(data_set='val', num_classes=classes)))
    lbann.contrib.lc.launcher.run(trainer, model, data_reader_proto, opt,
                                  job_name = 'lbann_alexnet',
                                  **kwargs)
