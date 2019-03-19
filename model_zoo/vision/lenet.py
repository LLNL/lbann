#!/usr/bin/env python3
import argparse
from os.path import abspath, dirname, join
import google.protobuf.text_format as txtf
import lbann.proto as lp
from lbann.models import LeNet
from lbann.proto import lbann_pb2

# Command-line arguments
data_reader_prototext = join(dirname(dirname(abspath(__file__))),
                             'data_readers',
                             'data_reader_mnist.prototext')
parser = argparse.ArgumentParser()
parser.add_argument(
    'file', nargs='?', default='model.prototext', type=str,
    help='exported prototext file')
parser.add_argument(
    '--mini-batch-size', action='store', default=64, type=int,
    help='mini-batch size (default: 64)', metavar='NUM')
parser.add_argument(
    '--num-epochs', action='store', default=100, type=int,
    help='number of epochs (default: 100)', metavar='NUM')
parser.add_argument(
    '--num-labels', action='store', default=10, type=int,
    help='number of data classes (default: 10)', metavar='NUM')
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
args = parser.parse_args()

# Construct layer graph
input = lp.Input()
images = lp.Identity(input)
labels = lp.Identity(input)
preds = LeNet(args.num_labels)(images)
softmax = lp.Softmax(preds)
loss = lp.CrossEntropy([softmax, labels])
acc = lp.CategoricalAccuracy([softmax, labels])

# Setup model
model = lp.Model(args.mini_batch_size,
                 args.num_epochs,
                 layers=lp.traverse_layer_graph(input),
                 objective_function=loss,
                 metrics=[lp.Metric(acc, name='accuracy', unit='%')],
                 callbacks=[lp.CallbackPrint(), lp.CallbackTimer()])

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

# Save to file
lp.save_prototext(args.file,
                  model=model, optimizer=opt,
                  data_reader=data_reader_proto)
