"""
Constructs the LBANN distributed training script for transformers.
"""
import argparse
import datetime
import os.path
from typing import Optional, Union

import lbann
import lbann.models
import lbann.contrib.args
import lbann.contrib.launcher
from lbann.launcher.batch_script import BatchScript
from lbann.util.data import Dataset, construct_python_dataset_reader
from lbann.launcher import make_timestamped_work_dir

import utils.paths
import parallelism


def construct_training_task(
        model: lbann.Model,
        args: argparse.Namespace,
        learning_rate: float = 0.0001,
        beta1: float = 0.9,
        beta2: float = 0.98,
        eps: float = 1e-9,
        clip_gradient: float = 0.0,
        lr_decay: str = 'fixed',
        lr_decay_steps: int = 0,
        end_learning_rate: float = 1e-5,
        warmup_steps: int = 0,
        adamw_decay: float = 0.1,
        dataset: Optional[Dataset] = None,
        validation_dataset: Optional[Dataset] = None,
        profiler_cmd: str = None,
        preamble_cmds: list = []) -> BatchScript:
    """
    Construct an LBANN trainer batch script for training transformers.

    :param model: An LBANN model.
    :param args: Command-line arguments.
    :param learning_rate: Learning rate.
    :param beta1: Adam beta1 factor.
    :param beta2: Adam beta2 factor.
    :param eps: Adam epsilon factor.
    :param clip_gradient: Clip gradient norm to value (0 disables).
    :param lr_decay: Learning rate decay type (values: fixed, cosine, none).
    :param lr_decay_steps: Steps for the total learning decay process (in cosine
                           decay).
    :param end_learning_rate: Learning rate after decay.
    :param warmup_steps: Learning rate warmup steps.
    :param adamw_decay: Weight decay if using the AdamW optimizer.
    :param profiler_cmd: profiler command and arguments to prepend to the lbann command.
    :param dataset: An optional Dataset object to use in training. If not given,
                    uses the ``--dataset`` argument string.
    :param validation_dataset: An optional Dataset object to use in validation.
                               Requires the ``dataset`` kwarg to be used.
    :return: A batch script object that will run distributed training.
    """

    # Setup working directory
    #timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    #work_dir = f'{timestamp}_{args.job_name}'
    #work_dir = os.path.abspath(work_dir)
    #os.makedirs(work_dir, exist_ok=True)

    nodes=args.nodes
    ppn=args.procs_per_node
    work_dir = make_timestamped_work_dir(job_name=args.job_name, nodes=nodes, procs_per_node=ppn)

    # Create batch script
    train_script = make_batch_script(
        model,
        dataset or args.dataset,
        work_dir,
        args,
        learning_rate,
        beta1,
        beta2,
        eps,
        clip_gradient,
        lr_decay,
        lr_decay_steps,
        end_learning_rate,
        warmup_steps,
        adamw_decay,
        validation_dataset=validation_dataset,
        profiler_cmd=profiler_cmd,
        preamble_cmds=preamble_cmds
    )

    return train_script


# ----------------------------------------------
# Data reader
# ----------------------------------------------
def make_data_reader(dataset_name: str, fraction: float, validate: bool,
                     val_fraction: float, always_shuffle: bool):
    reader = lbann.reader_pb2.DataReader()
    _reader = reader.reader.add()
    _reader.name = 'python'
    _reader.role = 'train'
    _reader.shuffle = (True if always_shuffle
                       or 'pretokenized' not in dataset_name else False)
    _reader.fraction_of_data_to_use = fraction
    _reader.python.module = dataset_name
    _reader.python.module_dir = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'datasets',
    )
    _reader.python.sample_function = 'get_train_sample'
    _reader.python.num_samples_function = 'num_train_samples'
    _reader.python.sample_dims_function = 'sample_dims'

    if validate:
        # Validation data reader
        vreader = reader.reader.add()
        vreader.name = 'python'
        vreader.role = 'validate'
        vreader.shuffle = False
        vreader.fraction_of_data_to_use = val_fraction
        vreader.python.module = _reader.python.module
        vreader.python.module_dir = _reader.python.module_dir
        vreader.python.sample_function = 'get_val_sample'
        vreader.python.num_samples_function = 'num_val_samples'
        vreader.python.sample_dims_function = 'sample_dims'

    return reader


# ----------------------------------------------
# Batch script
# ----------------------------------------------
def make_batch_script(model: lbann.Model,
                      dataset_or_dataset_name: Union[Dataset, str],
                      work_dir: str,
                      args: argparse.Namespace,
                      learning_rate: float = 0.0001,
                      beta1: float = 0.9,
                      beta2: float = 0.98,
                      eps: float = 1e-9,
                      clip_gradient: float = 0.0,
                      lr_decay: str = 'fixed',
                      lr_decay_steps: int = 0,
                      end_learning_rate: float = 1e-5,
                      warmup_steps: int = 0,
                      adamw_decay: float = 0.1,
                      validation_dataset: Optional[Dataset] = None,
                      profiler_cmd: str = None,
                      preamble_cmds: list = []):
    # Setup training algorithm
    algo = lbann.BatchedIterativeOptimizer("sgd", epoch_count=args.num_epochs)
    if hasattr(args, 'kfac') and args.kfac:
        algo = create_kfac_optimizer(algo, args)

    # Create LBANN trainer and data reader
    trainer = lbann.Trainer(mini_batch_size=args.mini_batch_size,
                            training_algo=algo,
                            random_seed=args.random_seed)

    if isinstance(dataset_or_dataset_name, str):
        assert validation_dataset is None
        reader = make_data_reader(dataset_or_dataset_name,
                                  args.dataset_fraction,
                                  not args.skip_validation,
                                  args.validation_set_fraction,
                                  args.always_shuffle)
    else:
        readers = [
            construct_python_dataset_reader(
                dataset_or_dataset_name,
                role='train',
                shuffle=args.always_shuffle,
                fraction_of_data_to_use=args.dataset_fraction,
                prefetch_factor=args.dataset_prefetch_factor),
        ]
        if validation_dataset is not None:
            readers.append(
                construct_python_dataset_reader(
                    validation_dataset,
                    role='validate',
                    shuffle=False,
                    validation_fraction=args.validation_set_fraction,
                    fraction_of_data_to_use=args.validation_set_fraction))

        reader = lbann.reader_pb2.DataReader(reader=readers)

    # Optimizer with learning rate schedule
    if args.optimizer.lower() == 'adamw':
        opt = lbann.Adam(learn_rate=learning_rate,
                         beta1=beta1,
                         beta2=beta2,
                         eps=eps,
                         adamw_weight_decay=adamw_decay)
    elif args.optimizer.lower() == 'adam':
        opt = lbann.Adam(learn_rate=learning_rate,
                         beta1=beta1,
                         beta2=beta2,
                         eps=eps)

    if lr_decay == 'fixed':
        if warmup_steps > 0:
            raise NotImplementedError(
                'Warmup not implemented with fixed learning rate')

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
    elif lr_decay == 'cosine':
        model.callbacks.append(
            lbann.CallbackCosineDecayLearningRate(
                lr_max=learning_rate,
                lr_min=end_learning_rate,
                decay_steps=lr_decay_steps,
                initial_warmup_learning_rate=end_learning_rate,
                warmup_steps=warmup_steps,
            ))

        print(f'Training schedule: warmup to LR={learning_rate:.6f} in '
              f'{warmup_steps} steps, cosine decay to '
              f'LR={end_learning_rate:.6f} in {lr_decay_steps} steps')

    if clip_gradient > 0:
        model.callbacks.append(
            lbann.CallbackClipGradientNorm(global_norm=True,
                                           value=clip_gradient))

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

    if args.validate_every > 0:
        model.callbacks.append(
            lbann.CallbackEvaluateProgress(batch_interval=args.validate_every))

    # Print a progress bar
    if args.progress:
        model.callbacks.append(
            lbann.CallbackProgressBar(
                newline_interval=args.pbar_newline_interval,
                print_mem_usage=True,
                moving_average_length=args.pbar_moving_avg,
                bar_width=args.pbar_width,
                scientific_notation=args.pbar_scientific))

    model.callbacks.extend(lbann.contrib.args.create_profile_callbacks(args))

    script_params = lbann.contrib.args.get_scheduler_kwargs(args)
    script_params['work_dir'] = work_dir
    script_params['job_name'] = args.job_name
    script_params['environment'] = {
        "LBANN_NO_INPLACE": 1,
        "LBANN_DISABLE_DISTCONV": 1,
    }
    script_params['preamble_cmds']=preamble_cmds
    script_params['scheduler']=args.scheduler

    # Set FSDP ranks, if given, by changing the trainer grid height
    if args.fsdp_ranks > 0:
        script_params['environment'][
            'LBANN_TRAINER_GRID_HEIGHT'] = args.fsdp_ranks

    save_text = args.save_prototext
    filename = 'experiment.prototext' if save_text else 'experiment.protobin'
    # Create Protobuf file
    protobuf_file = os.path.join(work_dir, filename)

    lbann.proto.save_prototext(protobuf_file,
                               binary=not save_text,
                               trainer=trainer,
                               model=model,
                               data_reader=reader,
                               optimizer=opt)

    # Create batch script
    script_params.pop('setup_only', None)  # Drop this argument.
    script = lbann.contrib.launcher.make_batch_script(**script_params)
    script.add_command('echo "Started training at $(date)"')

    lbann_command=[]
    if profiler_cmd is not None:
        lbann_command.append(profiler_cmd)
    lbann_command.append(lbann.lbann_exe())
    lbann_command.append(f'--prototext={protobuf_file}')
    script.add_parallel_command(lbann_command + 
            lbann.contrib.args.get_profile_args(args) + \
            parallelism.get_layer_parallel_args())

    script.add_command('status=$?')
    script.add_command('echo "Finished training at $(date)"')
    script.add_command('exit ${status}')
    return script


def add_training_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--skip-validation",
                        action="store_true",
                        default=False,
                        help="Do not run validation (default: false)")
    parser.add_argument(
        "--always-shuffle",
        action="store_true",
        default=False,
        help=
        "Always shuffle training dataset, even if pretokenized (default: false)"
    )
    parser.add_argument(
        "--validation-set-fraction",
        type=float,
        default=0.01,
        help="Fraction of the validation dataset to use (default: 0.001)")
    parser.add_argument(
        "--save-prototext",
        action="store_true",
        default=False,
        help="Save prototext experiment file instead of protobin (slower but "
        "debuggable) (default: false)")
    parser.add_argument("--validate-every",
                        type=int,
                        default=100,
                        help="Run validation every N steps (default: 100)")
    parser.add_argument('--random-seed',
                        action='store',
                        default=None,
                        type=int,
                        help=f'Set random seed explicitly',
                        metavar='NUM')
    parser.add_argument('--pbar-newline-interval',
                        type=int,
                        default=100,
                        help='Number of iterations in progress bar before '
                        'printing a newline (default: 100)')
    parser.add_argument('--pbar-width',
                        type=int,
                        default=30,
                        help='Progress bar width, if enabled (default: 30)')
    parser.add_argument('--pbar-moving-avg',
                        type=int,
                        default=10,
                        help='Progress bar iteration time moving average '
                        'length in iterations. Disable moving average with 1 '
                        '(default: 10)')
    parser.add_argument('--pbar-scientific',
                        action='store_true',
                        default=False,
                        help='Use scientific notation for objective value '
                        'printouts in progress bar (default: false)')


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
