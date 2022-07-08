import argparse
import math

import numpy as np

import lbann
import lbann.models
import lbann.contrib.args
import lbann.contrib.launcher
import lbann.modules as lm
from lbann.core.util import get_parallel_strategy_args


class CosmoFlow(lm.Module):
    """The CosmoFlow neural network.

    See:
        Amrita Mathuriya, Deborah Bard, Peter Mendygral, Lawrence Meadows,
        James Arnemann, Lei Shao, Siyu He, Tuomas Karna, Diana Moise,
        Simon J. Pennycook, Kristyn Maschhoff, Jason Sewall, Nalini Kumar,
        Shirley Ho, Michael F. Ringenburg, Prabhat, and Victor Lee.
        "Cosmoflow: Using deep learning to learn the universe at scale."
        Proceedings of the International Conference for High Performance
        Computing, Networking, Storage, and Analysis, SC'18, pp. 65:1-65:11,
        2018.
    """

    global_count = 0  # Static counter, used for default names

    def __init__(self,
                 input_width,
                 output_size,
                 name=None,
                 use_bn=False,
                 bn_statistics_group_size=None):
        """Initialize CosmFlow.

        Args:
            input_width (int): Size of each spatial dimension of input data.
            output_size (int): Size of output tensor.
            name (str, optional): Module name
                (default: 'cosmoflow_module<index>').
            use_bn (bool): Whether or not batch normalization layers are used.
            bn_statistics_group_size (int): The number of samples
                for each batch-normalization group.
        """

        CosmoFlow.global_count += 1
        self.instance = 0
        self.name = (name if name
                     else 'cosmoflow_module{0}'.format(CosmoFlow.global_count))
        self.input_width = input_width
        self.use_bn = use_bn

        assert self.input_width in [128, 256, 512]
        self.cp_params = [
            {"type": "conv", "out_channels": 16,  "kernel_size": 3, "stride": 1},
            {"type": "pool"},
            {"type": "conv", "out_channels": 32,  "kernel_size": 3, "stride": 1},
            {"type": "pool"},
            {"type": "conv", "out_channels": 64,  "kernel_size": 3, "stride": 1},
            {"type": "pool"},
            {"type": "conv", "out_channels": 128, "kernel_size": 3, "stride": 2},
            {"type": "pool"},
            {"type": "conv", "out_channels": 256, "kernel_size": 3, "stride": 1},
            {"type": "pool"},
            {"type": "conv", "out_channels": 256, "kernel_size": 3, "stride": 1},
            {"type": "conv", "out_channels": 256, "kernel_size": 3, "stride": 1},
        ]
        additional_pools = []
        if self.input_width == 256:
            additional_pools = [6]
        elif self.input_width == 512:
            additional_pools = [6, 7]

        for i in additional_pools:
            conv_idx = list(np.cumsum([1 if x["type"] == "conv" else
                                       0 for x in self.cp_params])).index(i)
            self.cp_params.insert(conv_idx+1, {"type": "pool"})

        for p in self.cp_params:
            if p["type"] == "conv":
                p["padding"] = int((p["kernel_size"]-1)/2)

        # Create convolutional blocks
        activation = lbann.LeakyRelu
        for i, param in enumerate(filter(lambda x: x["type"] == "conv", self.cp_params)):
            conv_name ="conv"+str(i+1)
            conv_weights = [lbann.Weights(
                initializer=lbann.GlorotUniformInitializer())]
            param_actual = dict(param)
            param_actual.pop("type", None)
            conv = ConvBNRelu(
                **param_actual,
                conv_weights=conv_weights,
                use_bn=self.use_bn,
                bn_statistics_group_size=bn_statistics_group_size,
                bn_zero_init=False,
                name=self.name+"_"+conv_name,
                activation=lbann.LeakyRelu)
            setattr(self, conv_name, conv)

        # Create fully-connected layers
        fc_params = [
            {"size": 2048},
            {"size": 256},
            {"size": output_size},
        ]
        for i, param in enumerate(fc_params):
            fc_name = "fc"+str(i+1)
            fc = lm.FullyConnectedModule(
                **param,
                activation=activation if i < len(fc_params)-1 else None,
                name=self.name+"_"+fc_name,
                weights=[lbann.Weights(initializer=lbann.GlorotUniformInitializer()),
                         lbann.Weights(initializer=lbann.ConstantInitializer(value=0.1))],
            )
            setattr(self, fc_name, fc)

    def forward(self, x):
        self.instance += 1

        def create_pooling(x, i):
            return lbann.Pooling(
                x, num_dims=3, has_vectors=False,
                pool_dims_i=3, pool_pads_i=1, pool_strides_i=2,
                pool_mode='average',
                name='{0}_pool{1}_instance{2}'.format(
                    self.name, i, self.instance))

        def create_dropout(x, i):
            return lbann.Dropout(
                x, keep_prob=0.8,
                name='{0}_drop{1}_instance{2}'.format(
                    self.name, i, self.instance))

        # Convolutional blocks
        i_conv = 1
        i_pool = 1
        for param in self.cp_params:
            if param["type"] == "conv":
                x = getattr(self, "conv{}".format(i_conv))(x)
                i_conv += 1

            else:
                x = create_pooling(x, i_pool)
                i_pool += 1

        # Fully-connected layers
        for i in range(3):
            x = create_dropout(x, i+1)
            x = getattr(self, "fc{}".format(i+1))(x)

        return x


class ConvBNRelu(lbann.modules.Module):
    """Convolution -> Batch normalization -> ReLU

    Adapted from ResNets. Assumes image data in NCDHW format.
    """

    def __init__(self, out_channels, kernel_size, stride, padding,
                 use_bn, bn_zero_init, bn_statistics_group_size,
                 activation, name,
                 conv_weights):
        """Initialize ConvBNRelu module.

        Args:
            out_channels (int): Number of output channels, i.e. number
                of convolution filters.
            kernel_size (int): Size of convolution kernel.
            stride (int): Convolution stride.
            padding (int): Convolution padding.
            use_bn (bool): Whether or not batch normalization layers are used.
            bn_zero_init (bool): Zero-initialize batch normalization
                scale.
            bn_statistics_group_size (int): Aggregation size for batch
                normalization statistics.
            activation (lbann.Layer): The activation function.
            name (str): Module name.
            conv_weights (lbann.Weights): Pre-defined weights.
        """

        super().__init__()
        self.name = name
        self.instance = 0
        self.bn_statistics_group_size = bn_statistics_group_size
        self.activation = activation
        self.use_bn = use_bn
        self.conv_weights = conv_weights

        # Initialize convolution
        self.conv = lbann.modules.Convolution3dModule(
            out_channels, kernel_size,
            stride=stride, padding=padding,
            bias=False, weights=self.conv_weights,
            name=self.name + '_conv')

        # Initialize batch normalization
        if self.use_bn:
            bn_scale_init = 0.0 if bn_zero_init else 1.0
            bn_scale = lbann.Weights(
                initializer=lbann.ConstantInitializer(value=bn_scale_init),
                name=self.name + '_bn_scale')
            bn_bias = lbann.Weights(
                initializer=lbann.ConstantInitializer(value=0.0),
                name=self.name + '_bn_bias')
            self.bn_weights = [bn_scale, bn_bias]

    def forward(self, x):
        self.instance += 1
        layer = self.conv(x)
        if self.use_bn:
            layer = lbann.BatchNormalization(
                layer, weights=self.bn_weights,
                statistics_group_size=self.bn_statistics_group_size,
                decay=0.999,
                name='{0}_bn_instance{1}'.format(
                    self.name, self.instance))
        if self.activation:
            layer = self.activation(
                layer,
                name='{0}_activation_instance{1}'.format(
                    self.name, self.instance))
        return layer


def create_cosmoflow_data_reader(
        train_path, val_path, test_path, num_responses):
    """Create a data reader for CosmoFlow.

    Args:
        {train, val, test}_path (str): Path to the corresponding dataset.
        num_responses (int): The number of parameters to predict.
    """

    reader_args = [
        {"role": "train", "data_filename": train_path},
        {"role": "validate", "data_filename": val_path},
        {"role": "test", "data_filename": test_path},
    ]

    for reader_arg in reader_args:
        reader_arg["data_file_pattern"] = "{}/*.hdf5".format(
            reader_arg["data_filename"])
        reader_arg["hdf5_key_data"] = "full"
        reader_arg["hdf5_key_responses"] = "unitPar"
        reader_arg["num_responses"] = num_responses
        reader_arg.pop("data_filename")

    readers = []
    for reader_arg in reader_args:
        reader = lbann.reader_pb2.Reader(
            name="hdf5",
            shuffle=(reader_arg["role"] != "test"),
            validation_percent=0,
            absolute_sample_count=0,
            percent_of_data_to_use=1.0,
            disable_labels=True,
            disable_responses=False,
            scaling_factor_int16=1.0,
            **reader_arg)

        readers.append(reader)

    return lbann.reader_pb2.DataReader(reader=readers)


if __name__ == "__main__":
    desc = ('Construct and run the CosmoFlow network on CosmoFlow dataset.'
            'Running the experiment is only supported on LC systems.')
    parser = argparse.ArgumentParser(description=desc)
    lbann.contrib.args.add_scheduler_arguments(parser)

    # General arguments
    parser.add_argument(
        '--job-name', action='store', default='lbann_cosmoflow', type=str,
        help='scheduler job name (default: lbann_cosmoflow)')
    parser.add_argument(
        '--mini-batch-size', action='store', default=1, type=int,
        help='mini-batch size (default: 1)', metavar='NUM')
    parser.add_argument(
        '--num-epochs', action='store', default=5, type=int,
        help='number of epochs (default: 100)', metavar='NUM')
    parser.add_argument(
        '--random-seed', action='store', default=None, type=int,
        help='the random seed (default: None)')

    # Model specific arguments
    parser.add_argument(
        '--input-width', action='store', default=128, type=int,
        help='the input spatial width (default: 128)')
    parser.add_argument(
        '--num-secrets', action='store', default=4, type=int,
        help='number of secrets (default: 4)')
    parser.add_argument(
        '--use-batchnorm', action='store_true',
        help='Use batch normalization layers')
    parser.add_argument(
        '--local-batchnorm', action='store_true',
        help='Use local batch normalization mode')
    default_lc_dataset = '/p/gpfs1/brainusr/datasets/cosmoflow/cosmoUniverse_2019_05_4parE/hdf5_transposed_dim128_float'
    for role in ['train', 'val', 'test']:
        default_dir = '{}/{}'.format(default_lc_dataset, role)
        parser.add_argument(
            '--{}-dir'.format(role), action='store', type=str,
            default=default_dir,
            help='the directory of the {} dataset'.format(role))

    # Parallelism arguments
    parser.add_argument(
        '--depth-groups', action='store', type=int, default=4,
        help='the number of processes for the depth dimension (default: 4)')
    parser.add_argument(
        '--depth-splits-pooling-id', action='store', type=int, default=None,
        help='the number of pooling layers from which depth_split is set (default: None)')
    parser.add_argument(
        '--gather-dropout-id', action='store', type=int, default=1,
        help='the number of dropout layers from which the network is gathered (default: 1)')

    parser.add_argument(
        '--dynamically-reclaim-error-signals', action='store_true',
        help='Allow LBANN to reclaim error signals buffers (default: False)')

    parser.add_argument(
        '--batch-job', action='store_true',
        help='Run as a batch job (default: false)')

    lbann.contrib.args.add_optimizer_arguments(
        parser,
        default_optimizer="adam",
        default_learning_rate=0.001,
    )
    args = parser.parse_args()

    if args.mini_batch_size * args.depth_groups < args.nodes * args.procs_per_node:
        print('WARNING the number of samples per mini-batch and depth group (partitions per sample)'
              ' is too small for the number of processes per trainer. Increasing the mini-batch size')
        args.mini_batch_size = int((args.nodes * args.procs_per_node) / args.depth_groups)
        print(f'Increasing mini_batch size to {args.mini_batch_size}')

    # Construct layer graph
    universes = lbann.Input(data_field='samples')
    secrets = lbann.Input(data_field='responses')
    statistics_group_size = 1 if args.local_batchnorm else -1
    preds = CosmoFlow(
        input_width=args.input_width,
        output_size=args.num_secrets,
        use_bn=args.use_batchnorm,
        bn_statistics_group_size=statistics_group_size)(universes)
    mse = lbann.MeanSquaredError([preds, secrets])
    obj = lbann.ObjectiveFunction([mse])
    layers = list(lbann.traverse_layer_graph([universes, secrets]))

    # Set parallel_strategy
    parallel_strategy = get_parallel_strategy_args(
        sample_groups=args.mini_batch_size,
        depth_groups=args.depth_groups)
    pooling_id = 0
    dropout_id = 0
    for i, layer in enumerate(layers):
        if layer == secrets:
            continue

        layer_name = layer.__class__.__name__
        if layer_name == 'Pooling':
            pooling_id += 1

            depth_splits_pooling_id = args.depth_splits_pooling_id
            if depth_splits_pooling_id is None:
                assert 2**math.log2(args.depth_groups) == args.depth_groups
                depth_splits_pooling_id = 5-(math.log2(args.depth_groups)-2)

            if pooling_id == depth_splits_pooling_id:
                parallel_strategy = dict(parallel_strategy.items())
                parallel_strategy['depth_splits'] = 1

        elif layer_name == 'Dropout':
            dropout_id += 1
            if dropout_id == args.gather_dropout_id:
                break

        layer.parallel_strategy = parallel_strategy

    # Set up model
    metrics = [lbann.Metric(mse, name='MSE', unit='')]
    callbacks = [
        lbann.CallbackPrint(),
        lbann.CallbackTimer(),
        lbann.CallbackGPUMemoryUsage(),
        lbann.CallbackDumpOutputs(
            directory='dump_acts/',
            layers=' '.join([preds.name, secrets.name]),
            execution_modes='test'
        ),
        lbann.CallbackProfiler(skip_init=True)]
    # # TODO: Use polynomial learning rate decay (https://github.com/LLNL/lbann/issues/1581)
    # callbacks.append(lbann.CallbackPolyLearningRate(
    #     power=1.0,
    #     num_epochs=100,
    #     end_lr=1e-7))
    model = lbann.Model(
        epochs=args.num_epochs,
        layers=layers,
        objective_function=obj,
        metrics=metrics,
        callbacks=callbacks
    )

    # Setup optimizer
    optimizer = lbann.contrib.args.create_optimizer(args)

    # Setup data reader
    data_reader = create_cosmoflow_data_reader(
        args.train_dir,
        args.val_dir,
        args.test_dir,
        num_responses=args.num_secrets)

    # Setup trainer
    random_seed_arg = {'random_seed': args.random_seed} \
        if args.random_seed is not None else {}
    trainer = lbann.Trainer(
        mini_batch_size=args.mini_batch_size,
        **random_seed_arg)

    # Runtime parameters/arguments
    environment = lbann.contrib.args.get_distconv_environment(
        num_io_partitions=args.depth_groups)
    if args.dynamically_reclaim_error_signals:
        environment['LBANN_KEEP_ERROR_SIGNALS'] = 0
    else:
        environment['LBANN_KEEP_ERROR_SIGNALS'] = 1
    lbann_args = ['--use_data_store']

    # Run experiment
    kwargs = lbann.contrib.args.get_scheduler_kwargs(args)
    lbann.contrib.launcher.run(
        trainer, model, data_reader, optimizer,
        job_name=args.job_name,
        environment=environment,
        lbann_args=lbann_args,
        batch_job=args.batch_job,
        **kwargs)
