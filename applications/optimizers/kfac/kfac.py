import argparse

import lbann.models
import lbann
import lbann.contrib.args
import lbann.contrib.launcher

# Use applications/vision/data
import os
import sys
vision_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                          "..", "..", "vision")
sys.path.append(vision_dir)
import data.mnist

from lbann.util import make_iterable

# ----------------------------------
# Command-line arguments
# ----------------------------------

DAMPING_PARAM_NAMES = ["act", "err", "bn_act", "bn_err"]

desc = ("Train a MLP or CNN on MNIST data with the K-FAC optimizer.")
parser = argparse.ArgumentParser(description=desc)
lbann.contrib.args.add_scheduler_arguments(parser, 'lbann_lenet_kfac')

# K-FAC configs.
parser.add_argument("--kfac", dest="kfac", action="store_const",
                    const=True, default=False,
                    help="use the K-FAC optimizer (default: false)")
parser.add_argument("--kfac-damping-warmup-steps", type=int, default=0,
                    help="the number of damping warmup steps")
parser.add_argument("--kfac-use-pi", dest="kfac_use_pi",
                    action="store_const",
                    const=True, default=False,
                    help="use the pi constant")
for n in DAMPING_PARAM_NAMES:
    parser.add_argument("--kfac-damping-{}".format(n), type=str, default="",
                        help="damping parameters for {}".format(n))
parser.add_argument("--kfac-update-interval-init", type=int, default=1,
                    help="the initial update interval of Kronecker factors")
parser.add_argument("--kfac-update-interval-target", type=int, default=1,
                    help="the target update interval of Kronecker factors")
parser.add_argument("--kfac-update-interval-steps", type=int, default=1,
                    help="the number of steps to interpolate -init and -target intervals")
parser.add_argument("--kfac-compute-interval-steps", type=int, default=1,
                    help="the number of steps after inverse matrices are calculated")
parser.add_argument("--use-eigen", dest="use_eigen",
                    action="store_const",
                    const=True, default=False)

# Job configs.
parser.add_argument("--batch-job", dest="batch_job",
                    action="store_const",
                    const=True, default=False,
                    help="submit a batch job")

# Training configs.
parser.add_argument("--num_epochs", type=int, default=20,
                    help="the number of epochs")
parser.add_argument("--mini_batch_size", type=int, default=100,
                    help="the mini-batch size")
parser.add_argument("--model", type=str,
                    choices=["mlp", "cnn", "gru"], default="mlp",
                    help="the model type (default: mlp)")

# Debugging configs.
parser.add_argument("--print-matrix", dest="print_matrix",
                    action="store_const",
                    const=True, default=False)
parser.add_argument("--print-matrix-summary", dest="print_matrix_summary",
                    action="store_const",
                    const=True, default=False)

lbann.contrib.args.add_optimizer_arguments(
    parser,
    default_optimizer="adam",
    default_learning_rate=0.0001,
)
args = parser.parse_args()

# ----------------------------------
# Construct layer graph
# ----------------------------------

# Setup data reader
data_reader = data.mnist.make_data_reader(validation_fraction=0)
num_classes = 10

# Input data
images = lbann.Input(data_field='samples')
labels = lbann.Input(data_field='labels')
has_bias = False
x = images

if args.model == "mlp":
    for i, num_neurons in enumerate([1000, 1000, num_classes]):
        if i:
            x = lbann.Relu(x)

        x = lbann.FullyConnected(
            x, num_neurons=num_neurons,
            has_bias=has_bias, name="ip{}".format(i+1),
            weights=[lbann.Weights(initializer=lbann.LeCunNormalInitializer())])

elif args.model == "cnn":
    for i, num_channels in enumerate([20, 50]):
        x = lbann.Convolution(
            x, num_dims=2, out_channels=num_channels,
            kernel_size=5, padding=0, stride=1,
            has_bias=has_bias,
            name="conv{}".format(i+1))
        x = lbann.Relu(x)
        x = lbann.Pooling(
            x, num_dims=2,
            pool_dims_i=2, pool_pads_i=0, pool_strides_i=2,
            pool_mode="max",
            name="pool{}".format(i+1))

    for i, num_neurons in enumerate([500, num_classes]):
        if i:
            x = lbann.Relu(x)

        x = lbann.FullyConnected(
            x, num_neurons=num_neurons,
            has_bias=has_bias, name="ip{}".format(i+1),
            weights=[lbann.Weights(initializer=lbann.LeCunNormalInitializer())])
elif args.model == "gru":
    zeros = lbann.Constant(
            value=0,
            num_neurons=[1, 28],
            name=f'_zeros'
        )
    x = lbann.Reshape(x, dims=[28, 28])
    x = lbann.GRU(x, zeros,hidden_size=28)
    x = lbann.Relu(x)
    x = lbann.FullyConnected(
            x, num_neurons=10,
            has_bias=has_bias, name="ip{}".format(0+1),
            weights=[lbann.Weights(initializer=lbann.LeCunNormalInitializer())])




probs = lbann.Softmax(x)

# Loss function and accuracy
loss = lbann.CrossEntropy(probs, labels)
acc = lbann.CategoricalAccuracy(probs, labels)
obj = lbann.ObjectiveFunction(loss)
metrics = [lbann.Metric(acc, name="accuracy", unit="%")]

# ----------------------------------
# Setup experiment
# ----------------------------------

# Setup callbacks
callbacks = [
    lbann.CallbackPrint(),
    lbann.CallbackTimer(),
    lbann.CallbackGPUMemoryUsage(),
]

# Setup training algorithm
algo = lbann.BatchedIterativeOptimizer("sgd", epoch_count=args.num_epochs)
if args.kfac:
    kfac_args = {}
    if args.kfac_use_pi:
        kfac_args["use_pi"] = 1
    if args.print_matrix:
        kfac_args["print_matrix"] = 1
    if args.print_matrix_summary:
        kfac_args["print_matrix_summary"] = 1
    for n in DAMPING_PARAM_NAMES:
        kfac_args["damping_{}".format(n)] = getattr(
            args, "kfac_damping_{}".format(n)).replace(",", " ")
    if args.kfac_damping_warmup_steps > 0:
        kfac_args["damping_warmup_steps"] = args.kfac_damping_warmup_steps
    if args.kfac_update_interval_init != 1 or args.kfac_update_interval_target != 1:
        kfac_args["update_intervals"] = "{} {}".format(
            args.kfac_update_interval_init,
            args.kfac_update_interval_target,
        )
    if args.kfac_update_interval_steps != 1:
        kfac_args["update_interval_steps"] = args.kfac_update_interval_steps
    kfac_args["compute_interval"] = args.kfac_compute_interval_steps
    kfac_args["use_eigen_decomposition"] = args.use_eigen
    algo = lbann.KFAC("kfac", algo, **kfac_args)

# Setup model
model = lbann.Model(
    args.num_epochs,
    layers=lbann.traverse_layer_graph([images,labels]),
    objective_function=obj,
    metrics=metrics,
    callbacks=callbacks)

# Setup optimizer
opt = lbann.contrib.args.create_optimizer(args)

# Setup trainer
trainer = lbann.Trainer(mini_batch_size=args.mini_batch_size, training_algo=algo)

# Setup environment variables
environment = {"LBANN_KEEP_ERROR_SIGNALS": 1}

# ----------------------------------
# Run experiment
# ----------------------------------
kwargs = lbann.contrib.args.get_scheduler_kwargs(args)
lbann.contrib.launcher.run(
    trainer, model, data_reader, opt,
    job_name=args.job_name,
    environment=environment,
    batch_job=args.batch_job,
    **kwargs)
