#!/usr/bin/env python3
import argparse
from os.path import dirname, join
import google.protobuf.text_format as txtf
import lbann
import lbann.models
import lbann.models.resnet
import lbann.proto
import lbann.contrib.args
import lbann.contrib.models.wide_resnet

# Default data reader
model_zoo_dir = dirname(dirname(__file__))
data_reader_prototext = join(model_zoo_dir,
                             'data_readers',
                             'data_reader_imagenet.prototext')

# Command-line arguments
desc = ('Construct and run ResNet on ImageNet-1K data. '
        'Running the experiment is only supported on LC systems.')
parser = argparse.ArgumentParser(description=desc)
lbann.contrib.args.add_scheduler_arguments(parser)
parser.add_argument(
    '--resnet', action='store', default=50, type=int,
    choices=(18, 34, 50, 101, 152),
    help='ResNet variant (default: 50)')
parser.add_argument(
    '--width', action='store', default=1, type=float,
    help='Wide ResNet width factor (default: 1)')
parser.add_argument(
    '--block-type', action='store', default=None, type=str,
    choices=('basic', 'bottleneck'),
    help='ResNet block type')
parser.add_argument(
    '--blocks', action='store', default=None, type=str,
    help='ResNet block counts (comma-separated list)')
parser.add_argument(
    '--block-channels', action='store', default=None, type=str,
    help='Internal channels in each ResNet block (comma-separated list)')
parser.add_argument(
    '--bn-stats-aggregation', action='store', default='local', type=str,
    help=('aggregation mode for batch normalization statistics '
          '(default: "local")'))
parser.add_argument(
    '--warmup', action='store_true', help='use a linear warmup')
parser.add_argument(
    '--mini-batch-size', action='store', default=256, type=int,
    help='mini-batch size (default: 256)', metavar='NUM')
parser.add_argument(
    '--num-epochs', action='store', default=90, type=int,
    help='number of epochs (default: 90)', metavar='NUM')
parser.add_argument(
    '--num-labels', action='store', default=1000, type=int,
    help='number of data classes (default: 1000)', metavar='NUM')
parser.add_argument(
    '--random-seed', action='store', default=0, type=int,
    help='random seed for LBANN RNGs', metavar='NUM')
lbann.contrib.args.add_optimizer_arguments(parser, default_learning_rate=0.1)
parser.add_argument(
    '--data-reader', action='store',
    default=data_reader_prototext, type=str,
    help='data reader prototext file (default: ' + data_reader_prototext + ')',
    metavar='FILE')
parser.add_argument(
    '--prototext', action='store', type=str,
    help='exported prototext file (do not run experiment)', metavar='FILE')
args = parser.parse_args()

# Due to a data reader limitation, the actual model realization must be
# hardcoded to 1000 labels for ImageNet.
imagenet_labels = 1000

# Choose ResNet variant
resnet_variant_dict = {18: lbann.models.ResNet18,
                       34: lbann.models.ResNet34,
                       50: lbann.models.ResNet50,
                       101: lbann.models.ResNet101,
                       152: lbann.models.ResNet152}
wide_resnet_variant_dict = {50: lbann.contrib.models.wide_resnet.WideResNet50_2}
block_variant_dict = {
    'basic': lbann.models.resnet.BasicBlock,
    'bottleneck': lbann.models.resnet.BottleneckBlock
}

if (any([args.block_type, args.blocks, args.block_channels])
    and not all([args.block_type, args.blocks, args.block_channels])):
    raise RuntimeError('Must specify all of --block-type, --blocks, --block-channels')
if args.block_type and args.blocks and args.block_channels:
    # Build custom ResNet.
    resnet = lbann.models.ResNet(
        block_variant_dict[args.block_type],
        imagenet_labels,
        list(map(int, args.blocks.split(','))),
        list(map(int, args.block_channels.split(','))),
        zero_init_residual=True,
        bn_stats_aggregation=args.bn_stats_aggregation,
        name='custom_resnet',
        width=args.width)
elif args.width == 1:
    # Vanilla ResNet.
    resnet = resnet_variant_dict[args.resnet](
        imagenet_labels,
        bn_stats_aggregation=args.bn_stats_aggregation)
elif args.width == 2 and args.resnet == 50:
    # Use pre-defined WRN-50-2.
    resnet = wide_resnet_variant_dict[args.resnet](
        imagenet_labels,
        bn_stats_aggregation=args.bn_stats_aggregation)
else:
    # Some other Wide ResNet.
    resnet = resnet_variant_dict[args.resnet](
        imagenet_labels,
        bn_stats_aggregation=args.bn_stats_aggregation,
        width=args.width)

# Construct layer graph
input = lbann.Input()
images = lbann.Identity(input)
labels = lbann.Identity(input)
preds = resnet(images)
probs = lbann.Softmax(preds)
cross_entropy = lbann.CrossEntropy([probs, labels])
top1 = lbann.CategoricalAccuracy([probs, labels])
top5 = lbann.TopKCategoricalAccuracy([probs, labels], k=5)
layers = list(lbann.traverse_layer_graph(input))

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
             lbann.CallbackTimer(),
             lbann.CallbackDropFixedLearningRate(
                 drop_epoch=[30, 60, 80], amt=0.1)]
if args.warmup:
    callbacks.append(
        lbann.CallbackLinearGrowthLearningRate(
            target=0.1 * args.mini_batch_size / 256, num_epochs=5))
model = lbann.Model(args.mini_batch_size,
                    args.num_epochs,
                    layers=layers,
                    objective_function=obj,
                    metrics=metrics,
                    callbacks=callbacks,
                    random_seed=args.random_seed)

# Setup optimizer
opt = lbann.contrib.args.create_optimizer(args)

# Load data reader from prototext
data_reader_proto = lbann.lbann_pb2.LbannPB()
with open(args.data_reader, 'r') as f:
  txtf.Merge(f.read(), data_reader_proto)
data_reader_proto = data_reader_proto.data_reader

# Save prototext
if args.prototext:
    lbann.proto.save_prototext(args.prototext,
                               model=model, optimizer=opt,
                               data_reader=data_reader_proto)

# Run experiment
if not args.prototext:
    from lbann.contrib.lc.paths import imagenet_dir, imagenet_labels
    import lbann.contrib.lc.launcher
    kwargs = lbann.contrib.args.get_scheduler_kwargs(args)
    classes = args.num_labels
    kwargs['lbann_args'] = (
        '--data_filedir_train={} --data_filename_train={} '
        '--data_filedir_test={} --data_filename_test={}'
        .format(imagenet_dir(data_set='train', num_classes=classes),
                imagenet_labels(data_set='train', num_classes=classes),
                imagenet_dir(data_set='val', num_classes=classes),
                imagenet_labels(data_set='val', num_classes=classes)))
    lbann.contrib.lc.launcher.run(model, data_reader_proto, opt,
                                  job_name='lbann_resnet',
                                  **kwargs)
