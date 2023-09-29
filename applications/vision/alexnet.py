import argparse
import lbann
import lbann.models
import lbann.contrib.args
import lbann.contrib.launcher
import data.imagenet

# Command-line arguments
desc = ('Construct and run AlexNet on ImageNet-1K data. '
        'Running the experiment is only supported on LC systems.')
parser = argparse.ArgumentParser(description=desc)
lbann.contrib.args.add_scheduler_arguments(parser, 'lbann_alexnet')
parser.add_argument(
    '--mini-batch-size', action='store', default=256, type=int,
    help='mini-batch size (default: 256)', metavar='NUM')
parser.add_argument(
    '--num-epochs', action='store', default=100, type=int,
    help='number of epochs (default: 100)', metavar='NUM')
parser.add_argument(
    '--num-classes', action='store', default=1000, type=int,
    help='number of ImageNet classes (default: 1000)', metavar='NUM')
parser.add_argument(
    '--synthetic', action='store_true', default=False,
    help='Use synthetic data')
lbann.contrib.args.add_optimizer_arguments(parser)
args = parser.parse_args()

# Due to a data reader limitation, the actual model realization must be
# hardcoded to 1000 labels for ImageNet.
imagenet_labels = 1000

# Construct layer graph
images = lbann.Input(data_field='samples')
labels = lbann.Input(data_field='labels')
preds = lbann.models.AlexNet(imagenet_labels)(images)
probs = lbann.Softmax(preds)
cross_entropy = lbann.CrossEntropy(probs, labels)
top1 = lbann.CategoricalAccuracy(probs, labels)
top5 = lbann.TopKCategoricalAccuracy(probs, labels, k=5)
layers = list(lbann.traverse_layer_graph([images, labels]))

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
model = lbann.Model(args.num_epochs,
                    layers=layers,
                    weights=weights,
                    objective_function=obj,
                    metrics=metrics,
                    callbacks=callbacks)

# Setup optimizer
opt = lbann.contrib.args.create_optimizer(args)

# Setup data reader
data_reader = data.imagenet.make_data_reader(
    num_classes=args.num_classes,
    synthetic=args.synthetic)

# Setup trainer
trainer = lbann.Trainer(mini_batch_size=args.mini_batch_size)

# Run experiment
kwargs = lbann.contrib.args.get_scheduler_kwargs(args)
lbann.contrib.launcher.run(trainer, model, data_reader, opt,
                           job_name=args.job_name,
                           **kwargs)
