import argparse
import lbann
import lbann.models
import lbann.contrib.args
import lbann.contrib.models.wide_resnet
import lbann.contrib.launcher
import data.cifar10
import data.imagenet

# Command-line arguments
desc = ('Construct and run ResNet on ImageNet-1K data. '
        'Running the experiment is only supported on LC systems.')
parser = argparse.ArgumentParser(description=desc)
lbann.contrib.args.add_scheduler_arguments(parser)
parser.add_argument(
    '--job-name', action='store', default='lbann_image_ae', type=str,
    help='scheduler job name (default: lbann_resnet)')
parser.add_argument(
    '--width', action='store', default=1, type=float,
    help='Wide ResNet width factor (default: 1)')
parser.add_argument(
    '--bn-statistics-group-size', action='store', default=1, type=int,
    help=('Group size for aggregating batch normalization statistics '
          '(default: 1)'))
parser.add_argument(
    '--warmup', action='store_true', help='use a linear warmup')
parser.add_argument(
    '--mini-batch-size', action='store', default=256, type=int,
    help='mini-batch size (default: 256)', metavar='NUM')
parser.add_argument(
    '--num-epochs', action='store', default=90, type=int,
    help='number of epochs (default: 90)', metavar='NUM')
parser.add_argument(
    '--num-classes', action='store', default=1000, type=int,
    help='number of ImageNet classes (default: 1000)', metavar='NUM')
parser.add_argument(
    '--random-seed', action='store', default=0, type=int,
    help='random seed for LBANN RNGs', metavar='NUM')
parser.add_argument(
    '--dataset', action='store', default='imagenet', type=str,
    help='dataset to use; \"cifar10\" or \"imagenet\"')
parser.add_argument(
    '--data-reader-percent', action='store',
    default=1.0, type=float,
    help='the percent of the data to use (default: 1.0)', metavar='NUM')
lbann.contrib.args.add_optimizer_arguments(parser, default_learning_rate=0.1)
args = parser.parse_args()

# Due to a data reader limitation, the actual model realization must be
# hardcoded to 1000 labels for ImageNet; 10 for CIFAR10.
dataset = args.dataset;
if dataset == 'imagenet':
    num_labels=1000
elif dataset == 'cifar10':
    num_labels=10
else:
    print("Dataset must be cifar10 or imagenet. Try again.")
    exit()

# Construct layer graph
input_ = lbann.Input(name='input')
image = lbann.Identity(input_, name='images')
dummy = lbann.Dummy(input_, name='labels')

# Encoder
encode1 = lbann.FullyConnected(image,
                               name="encode1",
                               data_layout="model_parallel",
                               num_neurons=1000,
                               has_bias=True)

relu1 = lbann.Relu(encode1, name="relu1", data_layout="model_parallel")

dropout1 = lbann.Dropout(relu1,
                         name="dropout1",
                         data_layout="model_parallel",
                         keep_prob=0.8)

decode1 = lbann.FullyConnected(dropout1,
                               name="decode1",
                               data_layout="model_parallel",
                               hint_layer="image",
                               has_bias=True)

reconstruction = lbann.Sigmoid(decode1,
                               name="reconstruction",
                               data_layout="model_parallel")

dropout2 = lbann.Dropout(reconstruction,
                         name="dropout2",
                         data_layout="model_parallel",
                         keep_prob=0.8)


# Reconstruction
mean_squared_error = lbann.MeanSquaredError([dropout2, image],
                             name="mean_squared_error")



preds = resnet(images)
probs = lbann.Softmax(preds)
cross_entropy = lbann.CrossEntropy(probs, labels)
top1 = lbann.CategoricalAccuracy(probs, labels, name='louise')
top5 = lbann.TopKCategoricalAccuracy(probs, labels, k=5)
layers = list(lbann.traverse_layer_graph(input_))

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

img_strategy = lbann.AutoencoderStrategy(
    input_layer_name='input',
    num_tracked_images=10)

img_dump_cb = lbann.CallbackSummarizeImages(
    selection_strategy=img_strategy,
    image_source_layer_name='images',
    epoch_interval=1)

callbacks = [lbann.CallbackPrint(),
             lbann.CallbackTimer(),
#             lbann.CallbackSummary(batch_interval = 2,
#                                   mat_interval = 3),
             lbann.CallbackDropFixedLearningRate(
                 drop_epoch=[30, 60, 80], amt=0.1),
             img_dump_cb]
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
                    random_seed=args.random_seed,
                    summary_dir="/g/g13/graham63/workspace/code/lbann")

# Setup optimizer
opt = lbann.contrib.args.create_optimizer(args)

# Setup data reader
num_classes=min(args.num_classes, num_labels)

if dataset == "cifar10":
    data_reader = data.cifar10.make_data_reader(num_classes=num_classes)
else:
    data_reader = data.imagenet.make_data_reader(num_classes=num_classes)

# Setup trainer
trainer = lbann.Trainer()

# Run experiment
kwargs = lbann.contrib.args.get_scheduler_kwargs(args)
kwargs['lbann_args'] = '--data_reader_percent='+str(args.data_reader_percent)
lbann.contrib.launcher.run(trainer, model, data_reader, opt,
                           job_name=args.job_name,
                           **kwargs)
