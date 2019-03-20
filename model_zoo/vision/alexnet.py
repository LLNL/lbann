#!/usr/bin/env python3
import argparse
from os.path import join
import google.protobuf.text_format as txtf
import lbann.proto as lp
from lbann.models import AlexNet
from lbann.proto import lbann_pb2
from lbann.utils import lbann_dir
import lbann.contrib.args

# Command-line arguments
desc = ('Construct and run AlexNet on MNIST data. '
        'Running the experiment is only supported on LC systems.')
data_reader_prototext = join(lbann_dir(),
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
parser.add_argument(
    '--optimizer', action='store', default='momentum', type=str,
    choices=('momentum', 'sgd', 'adam', 'adagrad', 'rmsprop'),
    help='optimizer (default: momentum)')
parser.add_argument(
    '--optimizer-learning-rate',
    action='store', default=0.01, type=float,
    help='optimizer learning rate (default: 0.01)', metavar='VAL')
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
input = lp.Input()
images = lp.Identity(input)
labels = lp.Identity(input)
preds = AlexNet(args.num_labels)(images)
softmax = lp.Softmax(preds)
ce = lp.CrossEntropy([softmax, labels])
top1 = lp.CategoricalAccuracy([softmax, labels])
top5 = lp.TopKCategoricalAccuracy([softmax, labels], k=5)
layers = list(lp.traverse_layer_graph(input))

# Setup objective function
weights = set()
for l in layers:
    weights.update(l.weights)
    l2_reg = lp.L2WeightRegularization(weights=weights, scale=5e-4)
    obj = lp.ObjectiveFunction([ce, l2_reg])

# Setup model
metrics = [lp.Metric(top1, name='top-1 accuracy', unit='%'),
           lp.Metric(top5, name='top-5 accuracy', unit='%')]
callbacks = [lp.CallbackPrint(),
             lp.CallbackTimer(),
             lp.CallbackDropFixedLearningRate(
                 drop_epoch=[20,40,60], amt=0.1)]
model = lp.Model(args.mini_batch_size,
                 args.num_epochs,
                 layers=layers,
                 weights=weights,
                 objective_function=obj,
                 metrics=metrics,
                 callbacks=callbacks)

# Setup optimizer
lr = args.optimizer_learning_rate
opt = lp.Optimizer()
if args.optimizer == 'momentum':
    opt = lp.SGD(learn_rate=lr, momentum=0.9)
elif args.optimizer == 'sgd':
    opt = lp.SGD(learn_rate=lr)
elif args.optimizer == 'adam':
    opt = lp.Adam(learn_rate=lr, beta1=0.9, beta2=0.99, eps=1e-8)
elif args.optimizer == 'adagrad':
    opt = lp.AdaGrad(learn_rate=lr, eps=1e-8)
elif args.optimizer == 'rmsprop':
    opt = lp.RMSprop(learn_rate=lr, decay_rate=0.99, eps=1e-8)

# Load data reader from prototext
data_reader_proto = lbann_pb2.LbannPB()
with open(args.data_reader, 'r') as f:
  txtf.Merge(f.read(), data_reader_proto)
data_reader_proto = data_reader_proto.data_reader

# Save prototext
if args.prototext:
    lp.save_prototext(args.prototext,
                      model=model, optimizer=opt,
                      data_reader=data_reader_proto)

# Run experiment
if not args.disable_run:
    from lbann.contrib.lc.paths import imagenet_dir, imagenet_labels
    from lbann.contrib.lc.launcher import run
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
    run(model, data_reader_proto, opt,
        job_name = 'lbann_alexnet',
        **kwargs)
