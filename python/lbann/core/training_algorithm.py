"""LBANN Training Algorithms.

These are methods for training neural network models in LBANN. If a
user does not explicitly specify a training algorithm in their
lbann.Trainer object, the default is equivalent to:

BatchedDescent('Some name', epoch_count=1)

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
            msg.max_batches = self.batch_count
            msg.max_epochs = self.epoch_count
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
                     keep_weights_names: list[str] = [],
                     exchange_hyperparameters: bool = False,
                     checkpoint_dir: str = None):
            """Construct a new exchange strategy.

            Args:
                strategy:
                  Which strategy to use (default: "checkpoint_binary").
                keep_weights_names:
                  A list of weights names that should not be exchanged.
                exchange_hyperparameters:
                  If True, exchange all optimizer state. Only applies to
                  the "sendrecv_weights" strategy.
                checkpoint_dir: A path to a directory for storing the
                  checkpoint files. Only applies to "checkpoint_file".
            """
            self.strategy = strategy
            self.exchange_hyperparams = exchange_hyperparameters
            self.keep_weights_names = make_iterable(keep_weights_names)
            self.checkpoint_dir = checkpoint_dir

        def export_proto(self):
            """Get a protobuf representation of this object."""

            ExchangeStrategyMsg = AlgoProto.RandomPairwiseExchange.ExchangeStrategy
            msg = ExchangeStrategyMsg()
            msg.weights_name.extend([n for n in self.keep_weights_names])
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

    def __init__(self, metric_name: str,
                 metric_strategy: int = MetricStrategy.LOWER_IS_BETTER,
                 exchange_strategy = ExchangeStrategy()):
        """Construct a new RandomPairwiseExchange metalearning strategy.

        Args:
            metric_name:
              The name of the metric to use for tournaments. Must be
              present in all models seen by this strategy.
            metric_strategy:
              Criterion for picking a tournament winner.
            exchange_strategy:
              The algorithm used for exchanging models.
        """

        self.metric_name = metric_name
        self.metric_strategy = metric_strategy
        self.exchange_strategy = exchange_strategy

    def export_proto(self):
        """Get a protobuf representation of this object."""

        msg = AlgoProto.RandomPairwiseExchange()
        msg.metric_name = self.metric_name
        msg.metric_strategy = self.metric_strategy
        msg.exchange_strategy.CopyFrom(self.exchange_strategy.export_proto())
        return msg

def get_default_training_algo():
    """Get a default training algorithm.

    Returns:
        A batched descent algorithm that will run for a single epoch.
    """
    return BatchedIterativeOptimizer("default sgd", epoch_count=1)
