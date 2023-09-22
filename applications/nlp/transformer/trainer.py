"""
Constructs the LBANN distributed training script for transformers.
"""
import argparse
import datetime
import os.path

import lbann
import lbann.models
import lbann.contrib.args
import lbann.contrib.launcher
from lbann.launcher.batch_script import BatchScript

import utils.paths


def construct_training_task(model: lbann.Model,
                            args: argparse.Namespace) -> BatchScript:
    """
    Construct an LBANN trainer batch script for training transformers.

    :param model: An LBANN model.
    :param dataset: A module pointing to the dataset to train on.
    :param args: Command-line arguments.
    :return: A batch script object that will run distributed training.
    """

    # Setup working directory
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    work_dir = os.path.join(utils.paths.root_dir(), 'experiments',
                            f'{timestamp}_{args.job_name}')
    os.makedirs(work_dir, exist_ok=True)

    # Create batch script
    train_script = make_batch_script(model, args.dataset, work_dir, args)

    return train_script


# ----------------------------------------------
# Data reader
# ----------------------------------------------
def make_data_reader(dataset_name, fraction):
    reader = lbann.reader_pb2.DataReader()
    _reader = reader.reader.add()
    _reader.name = 'python'
    _reader.role = 'train'
    _reader.shuffle = True
    _reader.fraction_of_data_to_use = fraction
    _reader.python.module = dataset_name
    _reader.python.module_dir = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'datasets',
    )
    _reader.python.sample_function = 'get_train_sample'
    _reader.python.num_samples_function = 'num_train_samples'
    _reader.python.sample_dims_function = 'sample_dims'
    return reader


# ----------------------------------------------
# Batch script
# ----------------------------------------------
def make_batch_script(model: lbann.Model, dataset_name: str, work_dir: str,
                      args: argparse.Namespace):
    # Setup training algorithm
    algo = lbann.BatchedIterativeOptimizer("sgd", epoch_count=args.num_epochs)
    if hasattr(args, 'kfac') and args.kfac:
        algo = create_kfac_optimizer(algo, args)

    # Create LBANN trainer and data reader
    trainer = lbann.Trainer(mini_batch_size=args.mini_batch_size,
                            training_algo=algo)
    reader = make_data_reader(dataset_name, args.dataset_fraction)

    # Optimizer with learning rate schedule
    if args.optimizer.lower() == 'adamw':
        opt = lbann.Adam(learn_rate=0.0001, beta1=0.9, beta2=0.98, eps=1e-9,
                         weight_decay=1e-2)
    elif args.optimizer.lower() == 'adam':
        # Note: Rough approximation of
        #   embed_dim^-0.5 * min(step^-0.5, step*warmup^-1.5)
        # with embed_dim=512 and warmup=4000.
        opt = lbann.Adam(learn_rate=0.0001, beta1=0.9, beta2=0.98, eps=1e-9)
        model.callbacks.append(
            lbann.CallbackDropFixedLearningRate(
                drop_epoch=[1],
                amt=2,
            ))
        model.callbacks.append(
            lbann.CallbackDropFixedLearningRate(
                drop_epoch=[2, 4, 8, 12],
                amt=0.75,
            ))

    # Checkpoint after every epoch
    if args.checkpoint:
        trainer.callbacks.append(
            lbann.CallbackCheckpoint(
                checkpoint_dir=os.path.join(work_dir, 'checkpoint'),
                checkpoint_epochs=1,
            ))

        # Dump weights after every epoch
        model.callbacks.append(
            lbann.CallbackDumpWeights(
                directory=os.path.join(work_dir, 'weights'),
                epoch_interval=1,
            ))

    # Print a progress bar
    if args.progress:
        model.callbacks.append(lbann.CallbackProgressBar())

    profiler = lbann.contrib.args.create_profile_callback(args)
    if profiler is not None:
        model.callbacks.append(profiler)

    script_params = lbann.contrib.args.get_scheduler_kwargs(args)
    script_params['work_dir'] = work_dir
    script_params['job_name'] = args.job_name
    script_params['environment'] = {
        'LBANN_USE_CUBLAS_TENSOR_OPS': 0,
        'LBANN_USE_CUDNN_TENSOR_OPS': 0,
        "LBANN_KEEP_ERROR_SIGNALS": 1
    }

    # Create Protobuf file
    protobuf_file = os.path.join(work_dir, 'experiment.prototext')

    lbann.proto.save_prototext(protobuf_file,
                               trainer=trainer,
                               model=model,
                               data_reader=reader,
                               optimizer=opt)

    # Create batch script
    script = lbann.contrib.launcher.make_batch_script(**script_params)
    script.add_command('echo "Started training at $(date)"')
    script.add_parallel_command([
        lbann.lbann_exe(),
        f'--prototext={protobuf_file}',
    ] + lbann.contrib.args.get_profile_args(args))
    script.add_command('status=$?')
    script.add_command('echo "Finished training at $(date)"')
    script.add_command('exit ${status}')
    return script


# ----------------------------------------------
# Second-order optimization functionality
# ----------------------------------------------
KFAC_DAMPING_PARAM_NAMES = ["act", "err", "bn_act", "bn_err"]


def add_kfac_arguments(parser):
    # KFAC configs
    parser.add_argument("--kfac",
                        dest="kfac",
                        action="store_const",
                        const=True,
                        default=False,
                        help="use the K-FAC optimizer (default: false)")

    parser.add_argument("--disable-BN",
                        dest="disBN",
                        action="store_const",
                        const=True,
                        default=False,
                        help="Disable KFAC for BN")

    parser.add_argument("--poly-lr",
                        dest="polyLR",
                        action="store_const",
                        const=True,
                        default=False,
                        help="Enable KFAC for BN")

    parser.add_argument("--poly-decay",
                        type=int,
                        default=11,
                        help="decay in poly LR scheduler (default: 11)")

    parser.add_argument("--mixup",
                        type=float,
                        default=0,
                        help="Data mixup (default: disabled)")

    parser.add_argument(
        "--momentum",
        type=float,
        default=2,
        help="momentum in SGD overides optimizer  (default: 2(false))")

    parser.add_argument(
        "--enable-distribute-compute",
        dest="enable_distribute_compute",
        action="store_const",
        const=True,
        default=False,
        help="Enable distributed compute of precondition gradients")
    parser.add_argument("--kfac-damping-warmup-steps",
                        type=int,
                        default=0,
                        help="the number of damping warmup steps")
    parser.add_argument("--kfac-use-pi",
                        dest="kfac_use_pi",
                        action="store_const",
                        const=True,
                        default=False,
                        help="use the pi constant")

    parser.add_argument(
        "--kfac-sgd-mix",
        type=str,
        default="",
        help=
        "alogrithm will be switched to KFAC at first given epoch then alternate  (default: use KFAC for all epochs)"
    )

    parser.add_argument(
        "--lr-list",
        type=str,
        default="",
        help="change lr accroding to interval in --kfac-sgd-mix")
    for n in KFAC_DAMPING_PARAM_NAMES:
        parser.add_argument("--kfac-damping-{}".format(n),
                            type=str,
                            default="",
                            help="damping parameters for {}".format(n))
    parser.add_argument(
        "--kfac-update-interval-init",
        type=int,
        default=1,
        help="the initial update interval of Kronecker factors")
    parser.add_argument("--kfac-update-interval-target",
                        type=int,
                        default=1,
                        help="the target update interval of Kronecker factors")
    parser.add_argument(
        "--kfac-update-interval-steps",
        type=int,
        default=1,
        help="the number of steps to interpolate -init and -target intervals")
    parser.add_argument(
        "--kfac-compute-interval-steps",
        type=int,
        default=1,
        help="the number of steps after inverse matrices are calculated")
    parser.add_argument("--use-eigen",
                        dest="use_eigen",
                        action="store_const",
                        const=True,
                        default=False)

    # Debugging configs.
    parser.add_argument("--print-matrix",
                        dest="print_matrix",
                        action="store_const",
                        const=True,
                        default=False)
    parser.add_argument("--print-matrix-summary",
                        dest="print_matrix_summary",
                        action="store_const",
                        const=True,
                        default=False)


def create_kfac_optimizer(original_algo: lbann.TrainingAlgorithm,
                          args: argparse.Namespace) -> lbann.TrainingAlgorithm:
    """
    Creates a second-order K-FAC optimizer for the training task.
    """
    kfac_args = {}
    if args.kfac_use_pi:
        kfac_args["use_pi"] = 1
    if args.print_matrix:
        kfac_args["print_matrix"] = 1
    if args.print_matrix_summary:
        kfac_args["print_matrix_summary"] = 1
    for n in KFAC_DAMPING_PARAM_NAMES:
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
    kfac_args[
        "distribute_precondition_compute"] = args.enable_distribute_compute
    kfac_args["disable_layers"] = "prediction_layer"
    kfac_args["use_eigen_decomposition"] = args.use_eigen
    kfac_args["kfac_use_interval"] = args.kfac_sgd_mix

    return lbann.KFAC("kfac", original_algo, **kfac_args)
