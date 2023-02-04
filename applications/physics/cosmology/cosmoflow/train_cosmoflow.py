import cosmoflow_model

import argparse

import numpy as np

import lbann.contrib.args
import lbann.contrib.launcher
from lbann.core.util import get_parallel_strategy_args

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
    default_lc_dataset = '/p/gpfs1/brainusr/datasets/cosmoflow/cosmoUniverse_2019_05_4parE/hdf5_transposed_dim128_float/batch8'
    for role in ['train', 'val', 'test']:
        default_dir = '{}/{}'.format(default_lc_dataset, role)
        parser.add_argument(
            '--{}-dir'.format(role), action='store', type=str,
            default=default_dir,
            help='the directory of the {} dataset'.format(role))

    # Parallelism arguments
    parser.add_argument(
        '--depth-groups', action='store', type=int, default=4,
        help='the k-way partitioning of the depth dimension (default: 4)')
    parser.add_argument(
        '--sample-groups', action='store', type=int, default=1,
        help='the k-way partitioning of the sample dimension (default: 1)')
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

    # Set parallel_strategy
    parallel_strategy = get_parallel_strategy_args(
        sample_groups=args.sample_groups,
        depth_groups=args.depth_groups)
    model = cosmoflow_model.construct_cosmoflow_model(parallel_strategy=parallel_strategy,
                                                      local_batchnorm=args.local_batchnorm,
                                                      input_width=args.input_width,
                                                      num_secrets=args.num_secrets,
                                                      use_batchnorm=args.use_batchnorm,
                                                      num_epochs=args.num_epochs,
                                                      depth_splits_pooling_id = args.depth_splits_pooling_id,
                                                      gather_dropout_id = args.gather_dropout_id)

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
        serialize_io=True,
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
