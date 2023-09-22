import argparse
import lbann
import lbann.contrib.args
import lbann.contrib.launcher
import data.imagenet

LOG = True

DAMPING_PARAM_NAMES = ["act", "err", "bn_act", "bn_err"]
def list2str(l):
    return ' '.join(l)



def log(string):
    if LOG:
        print(string)


# DenseNet #####################################################################
# See src/proto/lbann.proto for possible functions to call.
# See PyTorch DenseNet:
# https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py
# See "Densely Connected Convolutional Networks" by Huang et. al p.4
def densenet(statistics_group_size,
             version,
             cumulative_layer_num,
             images_node
             ):
    if version == 121:
        growth_rate = 32  # k in the paper
        layers_per_block = (6, 12, 24, 16)
        num_initial_features = 64
    elif version == 169:
        growth_rate = 32  # k in the paper
        layers_per_block = (6, 12, 32, 16)
        num_initial_features = 64
    elif version == 201:
        growth_rate = 32  # k in the paper
        layers_per_block = (6, 12, 48, 32)
        num_initial_features = 64
    elif version == 161:
        growth_rate = 48  # k in the paper
        layers_per_block = (96, 48, 36, 24)
        num_initial_features = 96
    else:
        raise Exception('Invalid version={v}.'.format(v=version))
    batch_norm_size = 4

    parent_node, cumulative_layer_num = initial_layer(
        statistics_group_size,
        cumulative_layer_num, images_node,
        num_initial_features)
    num_features = num_initial_features
    # Start counting dense blocks at 1.
    for current_block_num, num_layers in enumerate(layers_per_block, 1):
        parent_nodes, cumulative_layer_num = dense_block(
            statistics_group_size,
            cumulative_layer_num,
            parent_node,
            batch_norm_size=batch_norm_size,
            current_block_num=current_block_num,
            growth_rate=growth_rate,
            num_layers=num_layers,
            num_initial_channels=num_initial_features
        )
        # num_features += num_layers * growth_rate
        for node in parent_nodes[1:]:
            num_features += node.out_channels
        parent_node = lbann.Concatenation(parent_nodes)
        cumulative_layer_num += 1
        log('densenet Concatenation. cumulative_layer_num={n}'.format(
            b=current_block_num, n=cumulative_layer_num))
        if current_block_num != len(layers_per_block):
            parent_node, cumulative_layer_num = transition_layer(
                statistics_group_size,
                current_block_num,
                cumulative_layer_num,
                parent_node,
                # In Python 3, this is integer division.
                out_channels=num_features//2,
            )
            num_features //= 2

    batch_normalization_node = standard_batchnorm(statistics_group_size,
                                                  parent_node)
    cumulative_layer_num += 1
    log('densenet BatchNormalization. cumulative_layer_num={n}'.format(
        b=current_block_num, n=cumulative_layer_num))

    relu_node = lbann.Relu(batch_normalization_node)
    cumulative_layer_num += 1
    log('densenet Relu. cumulative_layer_num={n}'.format(
        b=current_block_num, n=cumulative_layer_num))

    probs = classification_layer(
        cumulative_layer_num,
        relu_node
    )
    return probs


def initial_layer(statistics_group_size,
                  cumulative_layer_num,
                  images_node,
                  num_initial_channels
                  ):
    # 7x7 conv, stride 2
    convolution_node = lbann.Convolution(
        images_node,
        kernel_size=7,
        padding=3,
        stride=2,
        has_bias=False,
        num_dims=2,
        out_channels=num_initial_channels
    )
    cumulative_layer_num += 1
    log('initial_layer Convolution. cumulative_layer_num={n}'.format(
        n=cumulative_layer_num))

    batch_normalization_node = standard_batchnorm(statistics_group_size,
                                                  convolution_node)
    cumulative_layer_num += 1
    log('initial_layer BatchNormalization. cumulative_layer_num={n}'.format(
        n=cumulative_layer_num))

    relu_node = lbann.Relu(batch_normalization_node)
    cumulative_layer_num += 1
    log('initial_layer Relu. cumulative_layer_num={n}'.format(
        n=cumulative_layer_num))

    # 3x3 max pool, stride 2
    pooling_node = lbann.Pooling(
        relu_node,
        num_dims=2,
        pool_dims_i=3,
        pool_mode='max',
        pool_pads_i=1,
        pool_strides_i=2
        )
    cumulative_layer_num += 1
    log('initial_layer Pooling. cumulative_layer_num={n}'.format(
        n=cumulative_layer_num))

    return pooling_node, cumulative_layer_num


def standard_batchnorm(statistics_group_size, parent_node):
    return lbann.BatchNormalization(
        parent_node,
        bias_init=0.0,
        decay=0.9,
        epsilon=1e-5,
        scale_init=1.0,
        statistics_group_size=statistics_group_size
    )


def dense_block(statistics_group_size,
                cumulative_layer_num,
                parent_node,
                batch_norm_size,
                current_block_num,
                growth_rate,
                num_layers,
                num_initial_channels
                ):
    parent_nodes = [parent_node]
    # Start counting dense layers at 1.
    for current_layer_num in range(1, num_layers + 1):
        # channels from before block + (each dense layer has k=growth_rate channels)
        num_input_channels = num_initial_channels + (current_layer_num - 1) * growth_rate
        print('num_input_channels={c}'.format(c=num_input_channels))
        parent_node, cumulative_layer_num = dense_layer(
            statistics_group_size,
            current_block_num,
            current_layer_num,
            cumulative_layer_num,
            parent_nodes,
            batch_norm_size=batch_norm_size,
            growth_rate=growth_rate
        )
        parent_nodes.append(parent_node)
    return parent_nodes, cumulative_layer_num


def dense_layer(statistics_group_size,
                current_block_num,
                current_layer_num,
                cumulative_layer_num,
                parent_nodes,
                batch_norm_size,
                growth_rate
                ):
    concatenation_node = lbann.Concatenation(parent_nodes)
    cumulative_layer_num += 1
    log('dense_block={b} dense_layer={l} Concatenation. cumulative_layer_num={n}'.format(
        b=current_block_num, l=current_layer_num, n=cumulative_layer_num))
    conv_block_1_node, cumulative_layer_num = conv_block(
        statistics_group_size,
        current_block_num,
        current_layer_num,
        cumulative_layer_num,
        concatenation_node,
        kernel_size=1,
        padding=0,
        out_channels=batch_norm_size * growth_rate
    )
    conv_block_2_node, cumulative_layer_num = conv_block(
        statistics_group_size,
        current_block_num,
        current_layer_num,
        cumulative_layer_num,
        conv_block_1_node,
        kernel_size=3,
        padding=1,
        out_channels=growth_rate
    )
    return conv_block_2_node, cumulative_layer_num


def conv_block(statistics_group_size,
               current_block_num,
               current_layer_num,
               cumulative_layer_num,
               parent_node,
               kernel_size,
               padding,
               out_channels
               ):
    batch_normalization_node = standard_batchnorm(statistics_group_size,
                                                  parent_node)
    cumulative_layer_num += 1
    log('dense_block={b} dense_layer={l} BatchNormalization. cumulative_layer_num={n}'.format(
        b=current_block_num, l=current_layer_num, n=cumulative_layer_num))

    relu_node = lbann.Relu(batch_normalization_node)
    cumulative_layer_num += 1
    log(
        'dense_block={b} dense_layer={l} Relu. cumulative_layer_num={n}'.format(
            b=current_block_num, l=current_layer_num, n=cumulative_layer_num))

    convolution_node = lbann.Convolution(
        relu_node,
        kernel_size=kernel_size,
        padding=padding,
        stride=1,
        has_bias=False,
        num_dims=2,
        out_channels=out_channels
    )
    cumulative_layer_num += 1
    log('dense_block={b} dense_layer={l} Convolution. cumulative_layer_num={n}'.format(
        b=current_block_num, l=current_layer_num, n=cumulative_layer_num))

    return convolution_node, cumulative_layer_num


def transition_layer(statistics_group_size,
                     current_block_num,
                     cumulative_layer_num,
                     parent_node,
                     out_channels
                     ):
    batch_normalization_node = standard_batchnorm(statistics_group_size,
                                                  parent_node)
    cumulative_layer_num += 1
    log('dense_block={b} > transition_layer BatchNormalization. cumulative_layer_num={n}'.format(
        b=current_block_num,  n=cumulative_layer_num))

    relu_node = lbann.Relu(batch_normalization_node)
    cumulative_layer_num += 1
    log('dense_block={b} > transition_layer Relu. cumulative_layer_num={n}'.format(
        b=current_block_num, n=cumulative_layer_num))

    convolution_node = lbann.Convolution(
        relu_node,
        kernel_size=1,
        padding=0,
        stride=1,
        has_bias=False,
        num_dims=2,
        out_channels=out_channels
    )
    cumulative_layer_num += 1
    log('dense_block={b} > transition_layer Convolution. cumulative_layer_num={n}'.format(
        b=current_block_num, n=cumulative_layer_num))

    # 2x2 average pool, stride 2
    pooling_node = lbann.Pooling(
        convolution_node,
        num_dims=2,
        pool_dims_i=2,
        pool_mode='average',
        pool_pads_i=0,
        pool_strides_i=2
    )
    cumulative_layer_num += 1
    log('dense_block={b} > transition_layer Pooling. cumulative_layer_num={n}'.format(
        b=current_block_num, n=cumulative_layer_num))

    return pooling_node, cumulative_layer_num


def classification_layer(cumulative_layer_num,
                         parent_node):
    # 7x7 global average pool
    pooling_node = lbann.Pooling(
        parent_node,
        num_dims=2,
        pool_dims_i=7,
        pool_mode='average',
        pool_pads_i=1,
        pool_strides_i=1
    )
    cumulative_layer_num += 1
    log('classification_layer Pooling. cumulative_layer_num={n}'.format(
        n=cumulative_layer_num))

    fully_connected_node = lbann.FullyConnected(
        pooling_node,
        num_neurons=1000,
        has_bias=False
    )
    cumulative_layer_num += 1
    log('classification_layer FullyConnected. cumulative_layer_num={n}'.format(
        n=cumulative_layer_num))

    probabilities = lbann.Softmax(fully_connected_node)
    return probabilities


# Helpful Functions ############################################################
def get_args():
    desc = ('Construct and run DenseNet on ImageNet data. '
            'Running the experiment is only supported on LC systems.')
    parser = argparse.ArgumentParser(description=desc)
    lbann.contrib.args.add_scheduler_arguments(parser, 'lbann_densenet')
    parser.add_argument(
        '--mini-batch-size', action='store', default=256, type=int,
        help='mini-batch size (default: 256)', metavar='NUM')
    parser.add_argument(
        '--num-epochs', action='store', default=90, type=int,
        help='number of epochs (default: 90)', metavar='NUM')
    parser.add_argument(
        '--num-classes', action='store', default=1000, type=int,
        help='number of ImageNet classes (default: 1000)', metavar='NUM')
    lbann.contrib.args.add_optimizer_arguments(
        parser,
        default_optimizer='sgd',
        default_learning_rate=0.1
    )
    parser.add_argument(
        '--setup_only', action='store_true',
        help='do not run experiment (e.g. if only the prototext is desired)')

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

    parser.add_argument("--model", type=int, default=169,
                        help="DenseNet model (default: 169)")

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
    args = parser.parse_args()
    return args


def set_up_experiment(args,
                      input_,
                      probs,
                      labels):
    algo = lbann.BatchedIterativeOptimizer("sgd", epoch_count=args.num_epochs)

    
    # Set up objective function
    cross_entropy = lbann.CrossEntropy([probs, labels])
    layers = list(lbann.traverse_layer_graph(input_))
    l2_reg_weights = set()

    bn_layers = ""
    for l in layers:
        if type(l) == lbann.Convolution or type(l) == lbann.FullyConnected:
            l2_reg_weights.update(l.weights)
        if type(l) == lbann.BatchNormalization:
            bn_layers += " " + l.name


    # scale = weight decay
    l2_reg = lbann.L2WeightRegularization(weights=l2_reg_weights, scale=1e-4)
    objective_function = lbann.ObjectiveFunction([cross_entropy, l2_reg])

    # Set up model
    top1 = lbann.CategoricalAccuracy([probs, labels])
    top5 = lbann.TopKCategoricalAccuracy([probs, labels], k=5)
    metrics = [lbann.Metric(top1, name='top-1 accuracy', unit='%'),
               lbann.Metric(top5, name='top-5 accuracy', unit='%')]
    callbacks = [lbann.CallbackPrint(),
                 lbann.CallbackTimer(),
                 lbann.CallbackDropFixedLearningRate(
                     drop_epoch=[30, 60], amt=0.1)]
    model = lbann.Model(args.num_epochs,
                        layers=layers,
                        objective_function=objective_function,
                        metrics=metrics,
                        callbacks=callbacks)

    # Set up data reader
    data_reader = data.imagenet.make_data_reader(num_classes=args.num_classes)

    fraction = 0.001 * 2 * (args.mini_batch_size / 16) * 2

    if (fraction > 1):
        data_reader.reader[0].fraction_of_data_to_use = 1.0
    else:
        data_reader.reader[0].fraction_of_data_to_use = fraction

    # Set up optimizer
    if args.optimizer == 'sgd':
        print('Creating sgd optimizer')
        optimizer = lbann.core.optimizer.SGD(
            learn_rate=args.optimizer_learning_rate,
            momentum=0.9,
            nesterov=True
        )
    else:
        optimizer = lbann.contrib.args.create_optimizer(args)

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

    # Setup trainer
    trainer = lbann.Trainer(mini_batch_size=args.mini_batch_size, training_algo=algo)

    return trainer, model, data_reader, optimizer


def run_experiment(args,
                   trainer,
                   model,
                   data_reader,
                   optimizer):
    # Note: Use `lbann.run` instead for non-LC systems.
    kwargs = lbann.contrib.args.get_scheduler_kwargs(args)
    lbann.contrib.launcher.run(trainer, model, data_reader, optimizer,
                               job_name=args.job_name,
                               environment = {
                              'LBANN_USE_CUBLAS_TENSOR_OPS' : 0,
                              'LBANN_USE_CUDNN_TENSOR_OPS' : 0,
                              "LBANN_KEEP_ERROR_SIGNALS": 1
                                },
                              lbann_args=" --use_data_store --preload_data_store --node_sizes_vary",
                               **kwargs)


# Main function ################################################################
def main():
    # ----------------------------------
    # Command-line arguments
    # ----------------------------------

    args = get_args()

    # ----------------------------------
    # Construct layer graph
    # ----------------------------------

    images = lbann.Input(data_field='samples')
    # Start counting cumulative layers at 1.
    cumulative_layer_num = 1
    log('Input(datum). cumulative_layer_num={n}'.format(n=cumulative_layer_num))
    labels = lbann.Input(data_field='labels')
    cumulative_layer_num += 1
    log('Input(labels). cumulative_layer_num={n}'.format(n=cumulative_layer_num))

    probs = densenet(1,
        args.model, cumulative_layer_num, images)



    # ----------------------------------
    # Setup experiment
    # ----------------------------------

    (trainer, model, data_reader_proto, optimizer) = set_up_experiment(
        args, [images, labels], probs, labels)

    # ----------------------------------
    # Run experiment
    # ----------------------------------

    run_experiment(args, trainer, model, data_reader_proto, optimizer)


if __name__ == '__main__':
    main()
