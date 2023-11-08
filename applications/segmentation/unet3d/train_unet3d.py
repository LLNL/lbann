import unet3d_model
import argparse

import lbann
import lbann.contrib.args
import lbann.contrib.launcher
from lbann.core.util import get_parallel_strategy_args


def create_unet3d_data_reader(train_dir, test_dir):
    readers = []
    for role, shuffle, role_dir in [
            ("train", True, train_dir),
            ("test", False, test_dir)]:
        if role_dir is None:
            continue

        readers.append(lbann.reader_pb2.Reader(
            name="hdf5",
            role=role,
            shuffle=shuffle,
            data_file_pattern="{}/*.hdf5".format(role_dir),
            validation_fraction=0,
            fraction_of_data_to_use=1.0,
            scaling_factor_int16=1.0,
            hdf5_key_data="volume",
            hdf5_key_labels="segmentation",
            hdf5_hyperslab_labels=True,
            disable_labels=False,
            disable_responses=True,
        ))

    return lbann.reader_pb2.DataReader(reader=readers)


def create_unet3d_optimizer(learn_rate):
    # TODO: This is a temporal optimizer copied from CosomoFlow.
    adam = lbann.Adam(
        learn_rate=learn_rate,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8)
    return adam


if __name__ == '__main__':
    desc = ('Construct and run the 3D U-Net on a 3D segmentation dataset.'
            'Running the experiment is only supported on LC systems.')
    parser = argparse.ArgumentParser(description=desc)
    lbann.contrib.args.add_scheduler_arguments(parser, 'lbann_unet3d')

    # General arguments
    parser.add_argument(
        '--mini-batch-size', action='store', default=1, type=int,
        help='mini-batch size (default: 1)', metavar='NUM')
    parser.add_argument(
        '--num-epochs', action='store', default=5, type=int,
        help='number of epochs (default: 100)', metavar='NUM')

    # Model specific arguments
    parser.add_argument(
        '--learning-rate', action='store', default=0.001, type=float,
        help='the initial learning rate (default: 0.001)')
    parser.add_argument(
        '--partition-level', action='store', default=4, type=int,
        help='the spatial partition level (default: 4)')
    # Parallelism arguments
    parser.add_argument(
        '--depth-groups', action='store', type=int, default=4,
        help='the k-way partitioning of the depth dimension (default: 4)')
    parser.add_argument(
        '--sample-groups', action='store', type=int, default=1,
        help='the k-way partitioning of the sample dimension (default: 1)')
    default_lc_dataset = '/p/vast1/lbann/datasets/LiTS/hdf5_dim128_float'
    default_train_dir = '{}/train'.format(default_lc_dataset)
    default_test_dir = '{}/test'.format(default_lc_dataset)
    parser.add_argument(
        '--train-dir', action='store', type=str, default=default_train_dir,
        help='the directory of the training dataset (default: \'{}\')'
        .format(default_train_dir))
    parser.add_argument(
        '--test-dir', action='store', type=str, default=default_test_dir,
        help='the directory of the test dataset (default: \'{}\')'
        .format(default_test_dir))

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

    parallel_strategy = get_parallel_strategy_args(
        sample_groups=args.sample_groups,
        depth_groups=args.depth_groups)

    model = unet3d_model.construct_unet3d_model(parallel_strategy=parallel_strategy,
                                                num_epochs=args.num_epochs)

    # Setup optimizer
    optimizer = lbann.contrib.args.create_optimizer(args)

    # Setup data reader
    data_reader = create_unet3d_data_reader(
        train_dir=args.train_dir,
        test_dir=args.test_dir)

    # Setup trainer
    trainer = lbann.Trainer(mini_batch_size=args.mini_batch_size)

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
