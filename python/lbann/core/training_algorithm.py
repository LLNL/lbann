"""LBANN Training Algorithms.

These are methods for training neural network models in LBANN. If a
user does not explicitly specify a training algorithm in their
lbann.Trainer object, the default (generated in C++) is equivalent to:

BatchedIterativeOptimizer('sgd_train', epoch_count=model.num_epochs)

Recall that the LBANN Python front-end (PFE) does *NOT* generate
meaningful python objects. They are simply vehicles for data waiting
to be serialized into the protobuf format expected as input to the
main LBANN driver. Thus, PFE classes should not be extended except
when exposing protobuf wrappers for new algorithms supported by the
LBANN library (or downstream libraries using the LBANN infrastructure).

"""

from __future__ import annotations

from lbann import training_algorithm_pb2 as AlgoProto
from lbann.util import make_iterable

class TrainingAlgorithm:
    """Base class for LBANN training algorithms"""
    def __init__(self, name: str):
        """Construct a training algorithm.

        Args:
            name: A user-defined name to identify this object in logs.
        """

        self.name = name

    def export_proto(self):
        """Get a protobuf representation of this object."""

        algo = AlgoProto.TrainingAlgorithm()
        algo.name = self.name
        algo.parameters.Pack(self.do_export_proto())
        return algo

    def do_export_proto(self):
        """Get a protobuf representation of this object.

        Must be implemented in derived classes.
        """
        raise NotImplementedError

class BatchedIterativeOptimizer(TrainingAlgorithm):
    """Batched gradient descent training algorithm.

    Because LBANN manages optimizers on a weights-specific basis, and
    those optimizers are (theoretically) allowed to vary from weights
    to weights, it would be inaccurate to describe this as "SGD"
    (which would also clash with the "Optimizer"
    subclass). Nonetheless, this is the well-loved first-order descent
    algorithm that seems good enough most of the time for most people
    (or so seems the general consensus).

    Also as a result of the way in which LBANN manages optimizers, the
    usage requirements for this class are quite sparse: only the
    stopping criteria are specified. The rest of the behavior of this
    class is specified in the model.
    """

    class StoppingCriteria:
        """Stopping criteria for an instance of BatchedIterativeOptimizer.

        BatchedIterativeOptimizer can be done to a finite batch count, a finite
        epoch count, or a fixed number of (double-precision) seconds
        (note: time-based stopping criteria is not publicly available
        yet; this is just a bit of foreshadowing).

        When used as a subalgorithm within a composite algorithm (see
        LTFB, below), this specifies the stopping criteria for each
        invocation. The algorithm stops when any of the criteria is
        satisfied.
        """

        def __init__(self, batch_count: int = 0, epoch_count: int = 0,
                     seconds: float = 0.):
            """Construct a new BatchedIterativeOptimizer stopping criteria.

            Args:
              batch_count: Number of minibatches.
              epoch_count: Number of epochs.
              seconds: Maximum training duration.
            """
            self.batch_count = batch_count
            self.epoch_count = epoch_count
            self.seconds = seconds

        def export_proto(self):
            """Get a protobuf representation of this object."""
            msg = AlgoProto.SGD.TerminationCriteria()
            if self.batch_count > 0:
                msg.max_batches = self.batch_count
            if self.epoch_count > 0:
                msg.max_epochs = self.epoch_count
            if self.seconds > 0:
                msg.max_seconds = self.seconds
            return msg

    def __init__(self, name: str, num_iterations: int = 0, epoch_count: int = 0,
                 max_seconds: float = 0.):
        """Construct a new BatchedIterativeOptimizer instance.

        Args:
            name: A user-defined name to identify this object in logs.
            num_iterations: Number of minibatches.
            epoch_count: Number of epochs.
            max_seconds: Maximum training duration (seconds)
        """
        self.name = name
        self.stopping = self.StoppingCriteria(batch_count=num_iterations,
                                              epoch_count=epoch_count,
                                              seconds=max_seconds)

    def do_export_proto(self):
        """Get a protobuf representation of this object."""
        params = AlgoProto.SGD()
        params.stopping_criteria.CopyFrom(self.stopping.export_proto())
        return params

class MetaLearningStrategy:
    """Base class for metalearning strategies for LTFB."""
    def __init__(self):
        pass

class LTFB(TrainingAlgorithm):
    """Livermore tournament fast-batch (LTFB) algorithm.

    This training algorithm is a simple composite training
    algorithm. The MPI universe is subdivided into several
    "trainers". Each trainer applies a local training algorithm. At
    the completion of local training, a metaheuristic ("metalearning"
    algorithm) is applied to select a new set of models to continue.

    The usage requirements for this training algorithm are a
    fully-specified local training algorithm and stopping criteria for
    the outer loop.
    """

    class StoppingCriteria:
        """Stopping criteria for LTFB"""
        def __init__(self, metalearning_steps: int = 1):
            """Construct a new LTFB stopping criteria object

            Args:
                metalearning_steps:
                  The number of outer-loop iterations.
            """
            self.metalearning_steps = metalearning_steps

        def export_proto(self):
            """Get a protobuf representation of this object."""
            msg = AlgoProto.LTFB.TerminationCriteria()
            msg.max_tournaments = self.metalearning_steps
            return msg

    def __init__(self, name: str, local_algo: TrainingAlgorithm,
                 metalearning: MetaLearningStrategy,
                 metalearning_steps: int = 1):
        """Construct a new LTFB algorithm.

        Args:
            name:
              A user-defined name to identify this object in logs.
            local_algo:
              The trainer-local algorithm to apply.
            metalearning:
              The metalearning strategy to apply after local training.
            metalearning_steps:
              The number of outer-loop iterations.
        """
        self.name = name
        self.metalearning = metalearning
        self.local_algo = local_algo
        self.stopping = self.StoppingCriteria(metalearning_steps=metalearning_steps)

    def do_export_proto(self):
        """Get a protobuf representation of this object."""
        params = AlgoProto.LTFB()
        params.stopping_criteria.CopyFrom(self.stopping.export_proto())
        params.meta_learning_strategy.Pack(self.metalearning.export_proto())
        params.local_training_algorithm.CopyFrom(self.local_algo.export_proto())
        return params

class RandomPairwiseExchange(MetaLearningStrategy):
    """The classic LTFB pairwise tourament metalearning strategy.

    This metalearning strategy is the original algorithm used for
    LTFB. After each local training step, all trainers in the LBANN
    environment pair off and have a "tournament" (if the number of
    trainers is odd, one competes with itself). The tournament is a
    simple comparison of a metric value that's read from the
    model. There is an assumption that a metric with the same name
    exists in all trainers' models. The tournaments are evaluated on a
    trainer-specific data set, so it is possible that each trainer in
    a pair will think a different model has won. The winning model
    survives and is either used as the initial guess for the next
    round of local training or returned to the caller.

    Since this algorithm relies on a metric with a given name being
    present in a model, each instance of this metalearning strategy
    (and, by extension, the training algorithm in which it is used)
    has an implicit limitation on the set of models to which it can be
    applied.

    """

    # Fake an enum, maybe?
    class MetricStrategy:
        LOWER_IS_BETTER: int = 0
        HIGHER_IS_BETTER: int = 1

    # This is supposed to go away. I don't want to make it any more
    # visible than this. In the same vein, I don't want more "class"
    # stuff for the different strategies.
    class ExchangeStrategy:
        """The algorithm for exchanging model data in RandomPairwiseExchange.

        WARNING: The fate of this class is under consideration. It
        would be good for it to converge to a single algorithm. Hence,
        no effort has been made here to mirror the C++ polymorphism in
        this Python wrapper.

        There are currently three strategies that are subtly different
        in the way they exchange model data.

        1. "checkpoint_binary": This is the default strategy. Entire
           models are serialized to binary as though doing a
           checkpoint. These serialized models are then sent over the
           wire and deserialized by the receiver. No assumptions are
           imposed on the model.

        2. "checkpoint_file": Models are serialized to disk as though
           doing a checkpoint. A barrier synchronization is used to
           guard against races on the file. The target process will
           unpack the serialized model from file after the barrier. No
           assumptions are imposed on the model.

        3. "sendrecv_weights": Model weights are exchanged
           individually. Some amount of optimizer state is also
           exchanged, with an option to exchange all optimizer data
           (when `exchange_hyperparameters=True`). This method
           (IMPLICITLY!!!) assumes that the weights objects to be
           exchanged appear in the same order in both the source and
           the target trainers' instances of the model. While
           extremely fragile hackery could produce other cases that
           happen to work, this essentially implies that the model
           topology should be homogenous across all trainers.

        """

        def __init__(self, strategy: str = "checkpoint_binary",
                     weights_names: list[str] = [],
                     exchange_hyperparameters: bool = False,
                     checkpoint_dir: str = None):
            """Construct a new exchange strategy.

            Args:
                strategy:
                  Which strategy to use (default: "checkpoint_binary").
                weights_names:
                  A list of weights names that should be exchanged.
                exchange_hyperparameters:
                  If True, exchange all optimizer state. Only applies to
                  the "sendrecv_weights" strategy.
                checkpoint_dir: A path to a directory for storing the
                  checkpoint files. Only applies to "checkpoint_file".
            """
            self.strategy = strategy
            self.exchange_hyperparameters = exchange_hyperparameters
            self.weights_names = make_iterable(weights_names)
            self.checkpoint_dir = checkpoint_dir

        def export_proto(self):
            """Get a protobuf representation of this object."""

            ExchangeStrategyMsg = AlgoProto.RandomPairwiseExchange.ExchangeStrategy
            msg = ExchangeStrategyMsg()
            msg.weights_name.extend([n for n in self.weights_names])
            if self.strategy == "checkpoint_binary":
                CheckpointBinaryMsg = ExchangeStrategyMsg.CheckpointBinary
                msg.checkpoint_binary.CopyFrom(CheckpointBinaryMsg())
            elif self.strategy == "checkpoint_file":
                if self.checkpoint_dir:
                    msg.checkpoint_file.checkpoint_dir = self.checkpoint_dir
                else:
                    raise Exception("Must provide checkpoint dir")
            elif self.strategy == "sendrecv_weights":
                msg.sendrecv_weights.exchange_hyperparameters = self.exchange_hyperparameters
            else:
                raise ValueError("Unknown strategy")
            return msg

    def __init__(self,
                 metric_strategies: dict[str,int] = {},
                 exchange_strategy = ExchangeStrategy()):
        """Construct a new RandomPairwiseExchange metalearning strategy.

        Args:
            metric_strategies:
              Map from metric name to the criterion for picking a winner
              with respect to this metric
            exchange_strategy:
              The algorithm used for exchanging models.
        """

        self.metric_strategies = metric_strategies
        self.exchange_strategy = exchange_strategy

    def export_proto(self):
        """Get a protobuf representation of this object."""

        msg = AlgoProto.RandomPairwiseExchange()
        for key, value in self.metric_strategies.items():
            msg.metric_name_strategy_map[key] = value

        msg.exchange_strategy.CopyFrom(self.exchange_strategy.export_proto())
        return msg

class KFAC(TrainingAlgorithm):

    def __init__(
            self,
            name: str,
            first_order_optimizer: BatchedIterativeOptimizer,
            damping_act : str = '',
            damping_err : str = '',
            damping_bn_act : str = '',
            damping_bn_err : str = '',
            damping_warmup_steps : int = 0,
            kronecker_decay : float = 0.,
            print_time : bool = False,
            print_matrix : bool = False,
            print_matrix_summary : bool = False,
            use_pi : bool = False,
            update_intervals : str = '',
            update_interval_steps : int = 0,
            inverse_strategy : str = '',
            disable_layers : str = '',
            learning_rate_factor : float = 0,
    ):
        self.name = name
        self.first_order_optimizer = first_order_optimizer
        self.damping_act = damping_act
        self.damping_err = damping_err
        self.damping_bn_act = damping_bn_act
        self.damping_bn_err = damping_bn_err
        self.damping_warmup_steps = damping_warmup_steps
        self.kronecker_decay = kronecker_decay
        self.print_time = print_time
        self.print_matrix = print_matrix
        self.print_matrix_summary = print_matrix_summary
        self.use_pi = use_pi
        self.update_intervals = update_intervals
        self.update_interval_steps = update_interval_steps
        self.inverse_strategy = inverse_strategy
        self.disable_layers = disable_layers
        self.learning_rate_factor = learning_rate_factor

    def do_export_proto(self):
        """Get a protobuf representation of this object."""
        params = AlgoProto.KFAC()
        first_order_optimizer_proto = self.first_order_optimizer.export_proto()
        first_order_optimizer_proto.parameters.Unpack(params.sgd)
        params.damping_act = self.damping_act
        params.damping_err = self.damping_err
        params.damping_bn_act = self.damping_bn_act
        params.damping_bn_err = self.damping_bn_err
        params.damping_warmup_steps = self.damping_warmup_steps
        params.kronecker_decay = self.kronecker_decay
        params.print_time = self.print_time
        params.print_matrix = self.print_matrix
        params.print_matrix_summary = self.print_matrix_summary
        params.use_pi = self.use_pi
        params.update_intervals = self.update_intervals
        params.update_interval_steps = self.update_interval_steps
        params.inverse_strategy = self.inverse_strategy
        params.disable_layers = self.disable_layers
        params.learning_rate_factor = self.learning_rate_factor
        return params
