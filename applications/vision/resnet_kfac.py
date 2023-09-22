import argparse
import lbann
import lbann.models
import lbann.models.resnet
import lbann.contrib.args
import lbann.contrib.models.wide_resnet
import lbann.contrib.launcher
import data.imagenet
import math

DAMPING_PARAM_NAMES = ["act", "err", "bn_act", "bn_err"]
def list2str(l):
    return ' '.join(l)


# Command-line arguments
desc = ('Construct and run ResNet on ImageNet-1K data. '
        'Running the experiment is only supported on LC systems.')
parser = argparse.ArgumentParser(description=desc)
lbann.contrib.args.add_scheduler_arguments(parser, 'lbann_resnet')
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
    '--bn-statistics-group-size', action='store', default=16, type=int,
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

# KFAC configs
parser.add_argument("--kfac", dest="kfac", action="store_const",
                const=True, default=False,
                help="use the K-FAC optimizer (default: false)")
parser.add_argument("--disable-BN", dest="disBN", action="store_const",
                const=True, default=False,
                help="Disable KFAC for BN")

parser.add_argument("--poly-lr", dest="polyLR", action="store_const",
                const=True, default=False,
                help="Enable KFAC for BN")

parser.add_argument("--poly-decay", type=int, default=11,
                    help="decay in poly LR scheduler (default: 11)")

parser.add_argument("--dropout", dest="add_dropout", action="store_const",
                const=True, default=False,
                help="Add dropout after input")

parser.add_argument("--dropout-keep-val", type=float, default=0.8,
                help="Keep value of dropout layer after input (default: 0.8)")

parser.add_argument("--label-smoothing", type=float, default=0,
                help="label smoothing (default: 0)")

parser.add_argument("--mixup", type=float, default=0,
                    help="Data mixup (default: disabled)")

parser.add_argument("--momentum", type=float, default=2,
                    help="momentum in SGD overides optimizer  (default: 2(false))")

parser.add_argument("--enable-distribute-compute", dest="enable_distribute_compute", action="store_const",
                const=True, default=False,
                help="Enable distributed compute of precondition gradients")
parser.add_argument("--kfac-damping-warmup-steps", type=int, default=0,
                    help="the number of damping warmup steps")
parser.add_argument("--kfac-use-pi", dest="kfac_use_pi",
                    action="store_const",
                    const=True, default=False,
                    help="use the pi constant")

parser.add_argument("--kfac-sgd-mix", type=str, default="",
                        help="alogrithm will be switched to KFAC at first given epoch then alternate  (default: use KFAC for all epochs)")

parser.add_argument("--lr-list", type=str, default="",
                        help="change lr accroding to interval in --kfac-sgd-mix")
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
# Debugging configs.
parser.add_argument("--print-matrix", dest="print_matrix",
                    action="store_const",
                    const=True, default=False)
parser.add_argument("--print-matrix-summary", dest="print_matrix_summary",
                    action="store_const",
                    const=True, default=False)

lbann.contrib.args.add_optimizer_arguments(parser, default_learning_rate=0.1)
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
        zero_init_residual=false,
        bn_statistics_group_size=args.bn_statistics_group_size,
        name='custom_resnet',
        width=args.width)
elif args.width == 1:
    # Vanilla ResNet.
    resnet = resnet_variant_dict[args.resnet](
        imagenet_labels,
        bn_statistics_group_size=args.bn_statistics_group_size,zero_init_residual=False)
elif args.width == 2 and args.resnet == 50:
    # Use pre-defined WRN-50-2.
    resnet = wide_resnet_variant_dict[args.resnet](
        imagenet_labels,
        bn_statistics_group_size=args.bn_statistics_group_size)
else:
    # Some other Wide ResNet.
    resnet = resnet_variant_dict[args.resnet](
        imagenet_labels,
        bn_statistics_group_size=args.bn_statistics_group_size,
        width=args.width)

if(len(args.lr_list) > 0 and len(args.kfac_sgd_mix)==0):
    print("--lr-list should only be used with --kfac-sgd-mix")
    exit()


# Construct layer graph
images = lbann.Input(data_field='samples')
labels = lbann.Input(data_field='labels')

if(args.add_dropout):
    images = lbann.Dropout(
                images, keep_prob=args.dropout_keep_val,
                name="input_dropout")

preds = resnet(images)
probs = lbann.Softmax(preds)

if args.label_smoothing > 0:
    uniform_label = lbann.Constant(
            value=1/args.num_classes,
            num_neurons=[args.num_classes])
    labels = lbann.WeightedSum(
                labels,
                uniform_label,
                scaling_factors=[1-args.label_smoothing, args.label_smoothing])

cross_entropy = lbann.CrossEntropy(probs, labels)
top1 = lbann.CategoricalAccuracy(probs, labels)
top5 = lbann.TopKCategoricalAccuracy(probs, labels, k=5)
layers = list(lbann.traverse_layer_graph([images, labels]))

# Setup tensor core operations (just to demonstrate enum usage)
tensor_ops_mode = lbann.ConvTensorOpsMode.NO_TENSOR_OPS
for l in layers:
    if type(l) == lbann.Convolution:
        l.conv_tensor_op_mode=tensor_ops_mode

# Setup objective function
l2_reg_weights = set()
for l in layers:
    if type(l) == lbann.Convolution or type(l) == lbann.FullyConnected:
        l2_reg_weights.update(l.weights)

bn_layers = ""
for l in layers:
    if("bn" in l.name):
        bn_layers += " " + l.name
print(bn_layers)
l2_reg = lbann.L2WeightRegularization(weights=l2_reg_weights, scale=0.00005)
obj = lbann.ObjectiveFunction([cross_entropy, l2_reg])

# Setup model
metrics = [lbann.Metric(top1, name='top-1 accuracy', unit='%')]
# callbacks = [lbann.CallbackPrint(),
#              lbann.CallbackTimer(),
#              lbann.CallbackDropFixedLearningRate(
#                  drop_epoch=[5, 10, 15, 20, 25,  30, 35 ,40, 45, 50, 55, 60, 70, 80], amt=0.5)]

# callbacks = [lbann.CallbackPrint(),
#              lbann.CallbackTimer(),
#              lbann.CallbackDropFixedLearningRate(
#                  drop_epoch=[3,6,9,12,15,18,21,24,27,30,33,36,39,42], amt=0.5)]
if (args.polyLR and len(args.lr_list)==0):
    iterations = math.ceil(1281167 / args.mini_batch_size)
    # callbacks = [lbann.CallbackPrint(),
    #              lbann.CallbackTimer(),
    #              lbann.CallbackPolyLearningRate(
    #                  power=args.poly_decay, num_epochs=55, max_iter= 55*iterations )]

    callbacks = [lbann.CallbackPrint(),
                 lbann.CallbackTimer(),
                 lbann.CallbackPolyLearningRate(
                     power=args.poly_decay, num_epochs=100, max_iter= 100*iterations )]

elif (len(args.lr_list)>0):
    init_lr = args.optimizer_learning_rate
    lr_list = args.lr_list.split(" ")
    epoch_list = args.kfac_sgd_mix.split(" ")

    callbacks = [lbann.CallbackPrint(),
                 lbann.CallbackTimer()]

    for i in range(len(lr_list)):
        # factor_div =  float(lr_list[i]) / init_lr
        callbacks.append(lbann.CallbackSetLearningRate(step=int(epoch_list[i]), val=float(lr_list[i])))

else:

    callbacks = [lbann.CallbackPrint(),
                 lbann.CallbackTimer(),
                 lbann.CallbackDropFixedLearningRate(
                     drop_epoch=[20, 30, 35], amt=0.1)]

if args.mixup>0:
    None
    # callbacks.append(lbann.CallbackMixup(alpha=args.mixup, layers=images.name))

if args.warmup:
    callbacks.append(
        lbann.CallbackLinearGrowthLearningRate(
            target=0.1 , num_epochs=5))
model = lbann.Model(args.num_epochs,
                    layers=layers,
                    objective_function=obj,
                    metrics=metrics,
                    callbacks=callbacks)

# Setup optimizer
if(args.momentum < 1):
    opt = lbann.core.optimizer.SGD(learn_rate=args.optimizer_learning_rate, momentum=args.momentum)
else:
    opt = lbann.contrib.args.create_optimizer(args)

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
        kfac_args["kronecker_decay"] = 0.95
        kfac_args["compute_interval"] = args.kfac_compute_interval_steps
        kfac_args["distribute_precondition_compute"] = args.enable_distribute_compute
        kfac_args["disable_layers"]="molvae_module1_disc0_fc0_instance1_fc molvae_module1_disc0_fc0_instance2_fc"
        kfac_args["use_eigen_decomposition"] = args.use_eigen
        kfac_args["kfac_use_interval"] = args.kfac_sgd_mix

        print(args.kfac_sgd_mix)

        if args.disBN:
            kfac_args["disable_layers"]=bn_layers
        algo = lbann.KFAC("kfac", algo, **kfac_args)

# Setup data reader
data_reader = data.imagenet.make_data_reader(num_classes=args.num_classes)

# Setup trainer
trainer = lbann.Trainer(mini_batch_size=args.mini_batch_size, random_seed=args.random_seed, training_algo=algo)
#lbann_args=" --use_data_store --preload_data_store"
# Run experiment
kwargs = lbann.contrib.args.get_scheduler_kwargs(args)
lbann.contrib.launcher.run(trainer, model, data_reader, opt,
                           job_name=args.job_name,
                           environment = {
                              'LBANN_USE_CUBLAS_TENSOR_OPS' : 0,
                              'LBANN_USE_CUDNN_TENSOR_OPS' : 0,
                              "LBANN_KEEP_ERROR_SIGNALS": 1
                          },
                          lbann_args=" --use_data_store --preload_data_store --node_sizes_vary",
                           **kwargs)
