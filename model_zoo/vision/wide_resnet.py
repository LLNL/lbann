#!/usr/bin/env python3
import argparse
import os.path
import google.protobuf.text_format as txtf
import lbann
import lbann.modules
import lbann.contrib.args

# ----------------------------------
# Command-line arguments
# ----------------------------------

desc = ('Construct and run WRN-50-2 on ImageNet-1K data. '
        'Running the experiment is only supported on LC systems.')
data_reader_prototext = os.path.join(lbann.lbann_dir(),
                                     'model_zoo',
                                     'data_readers',
                                     'data_reader_imagenet.prototext')
parser = argparse.ArgumentParser(description=desc)
lbann.contrib.args.add_scheduler_arguments(parser)
parser.add_argument(
    '--bn-stats-aggregation', action='store', default='local', type=str,
    help=('aggregation mode for batch normalization statistics '
          '(default: "local")'))
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

# ----------------------------------
# Construct layer graph
# ----------------------------------

def make_block(input, in_channels, mid_channels, out_channels, stride):
    """Bottleneck residual block."""

    # Convolution branch
    y1 = input
    y1 = lbann.modules.Convolution2dModule(mid_channels, 1, bias=False)(y1)
    y1 = lbann.BatchNormalization(y1, stats_aggregation=args.bn_stats_aggregation)
    y1 = lbann.Relu(y1)
    y1 = lbann.modules.Convolution2dModule(mid_channels, 3,
                                           stride=stride,
                                           padding=1,
                                           bias=False)(y1)
    y1 = lbann.BatchNormalization(y1, stats_aggregation=args.bn_stats_aggregation)
    y1 = lbann.Relu(y1)
    y1 = lbann.modules.Convolution2dModule(out_channels, 1, bias=False)(y1)
    y1 = lbann.BatchNormalization(y1, stats_aggregation=args.bn_stats_aggregation)

    # Shortcut branch
    y2 = input
    if in_channels != out_channels or stride != 1:
        y2 = lbann.modules.Convolution2dModule(out_channels, 1,
                                               stride=stride,
                                               bias=False)(y2)
        y2 = lbann.BatchNormalization(y2, stats_aggregation=args.bn_stats_aggregation)

    # Output is sum of convolution and shortcut branches
    return lbann.Relu(lbann.Add([y1, y2]))

def make_group(input, in_channels, mid_channels, out_channels, stride, num_blocks):
    """Stacked residual blocks."""
    y = input
    for block in range(num_blocks):
        y = make_block(y,
                       in_channels if block == 0 else out_channels,
                       mid_channels,
                       out_channels,
                       stride)
    return y

# Input data
input = lbann.Input()
images = lbann.Identity(input)
labels = lbann.Identity(input)

# WRN-50-2-bottleneck
x = lbann.modules.Convolution2dModule(64, 7,
                                      bias=False,
                                      stride=2,
                                      padding=3)(images)
x = lbann.BatchNormalization(x, stats_aggregation=args.bn_stats_aggregation)
x = lbann.Relu(x)
x = lbann.Pooling(x, num_dims=2, has_vectors=False,
                  pool_dims_i=3, pool_pads_i=1, pool_strides_i=2,
                  pool_mode='max')
x = make_group(x, 64, 128, 256, 1, 3)
x = make_group(x, 256, 256, 512, 2, 4)
x = make_group(x, 512, 512, 1024, 2, 6)
x = make_group(x, 1024, 1024, 2048, 2, 3)
x = lbann.ChannelwiseMean(x)
x = lbann.modules.FullyConnectedModule(1000)(x)

# Evaluation
probs = lbann.Softmax(x)
cross_entropy = lbann.CrossEntropy([probs, labels])
top1 = lbann.CategoricalAccuracy([probs, labels])
top5 = lbann.TopKCategoricalAccuracy([probs, labels], k=5)

# Get list of layers
layers = list(lbann.traverse_layer_graph(input))

# ----------------------------------
# Setup experiment
# ----------------------------------

# Setup objective function
l2_reg_weights = set()
for l in layers:
    if type(l) == lbann.Convolution or type(l) == lbann.FullyConnected:
        l2_reg_weights.update(l.weights)
l2_reg = lbann.L2WeightRegularization(weights=l2_reg_weights, scale=1e-4)
obj = lbann.ObjectiveFunction([cross_entropy, l2_reg])

# Setup model
metrics = [lbann.Metric(top1, name='top-1 accuracy', unit='%'),
           lbann.Metric(top5, name='top-5 accuracy', unit='%')]
callbacks = [lbann.CallbackPrint(),
             lbann.CallbackTimer()]
model = lbann.Model(args.mini_batch_size,
                    args.num_epochs,
                    layers=layers,
                    objective_function=obj,
                    metrics=metrics,
                    callbacks=callbacks)

# Setup optimizer
opt = lbann.contrib.args.create_optimizer(args)

# Load data reader from prototext
# Load data reader from prototext
data_reader_proto = lbann.lbann_pb2.LbannPB()
with open(args.data_reader, 'r') as f:
  txtf.Merge(f.read(), data_reader_proto)
data_reader_proto = data_reader_proto.data_reader

# Save prototext if needed
if args.prototext:
    lbann.proto.save_prototext(args.prototext,
                               model=model, optimizer=opt,
                               data_reader=data_reader_proto)

# ----------------------------------
# Run experiment
# ----------------------------------
# Note: Use `lbann.run` instead for non-LC systems.

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
    lbann.contrib.lc.launcher.run(model, data_reader_proto, opt,
                                  job_name = 'lbann_wide_resnet',
                                  **kwargs)
