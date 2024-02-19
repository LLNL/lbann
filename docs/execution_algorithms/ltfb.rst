.. role:: python(code)
          :language: python

============================================================
Livermore Tournament Fast Batch (LTFB)
============================================================

The "Livermore Tournament Fast Batch" (LTFB) algorithm is a
metaheuristic method for training neural networks.

The distributed computing environment is partitioned into several
"trainers", each of which is capable of independently training a
neural network model. The algorithm, then, is as follows. Each trainer
executes a (potentially unique) training algorithm on its local model
with a local partition of the training set. Then all the trainers
enter a "metalearning" step to provide additional modification. After
the final metalearning step, an additional round of local training is
applied to further diversify the returned model set. Each trainer in
the trainer set returns a possibly unique model. In a global sense,
this algorithm can be thought to take :code:`num_trainers` model
descriptions as inputs and returns :code:`num_trainers` trained
models.

The LBANN framework decomposes the algorithm into distinct strategies,
one for local training of models and one for applying the metalearning
heuristic.

The local training algorithm is a trainer-local strategy
for evolving a trainer's instance of a model. As the name suggests,
the local algorithm should be local to the instance of the
trainer. That is, it should not interact with the training algorithm
running in another trainer (though this is not strongly enforced). The
local algorithm used by a given trainer need not match the type,
(hyper)parameters, or stopping criteria of the local algorithm used by
any other trainer.

The metalearning strategy is assumed to be a global operation across
the LBANN "universe" (i.e., the collection of all trainers running in
a single job). The canonical implementation of the LTFB algorithm will
use a random pairwise tournament metalearning strategy in which the
trainers in the universe are paired off, with each set of partners
exchanging models, evaluating them using a trainer-local "tournament"
data set, and making a trainer-local decision about which of the two
candidate models "wins" the tournament and survives. Because the
tournament winner is decided locally by each trainer in the pair, the
two trainers need not agree on the tournament winner. Winning models
survive the tournament rounds and enter more local training, while the
losing models are forgotten as training continues.

The sole stopping criterion for the LTFB algorithm is the number of
metalearning steps to perform. Recall that one additional instance of
local training will occur after the last metalearning step. It is
critical that each trainer in the universe specifies the same number
of metalearning steps. For the random pairwise exchange strategy,
passing different LTFB stopping criteria to each trainer could cause a
hang if a non-participating trainer is selected as the partner of a
participating trainer in the tournament round -- the exchange algorithm
has no way to test whether a given trainer is or is not participating
in the tournament.

----------------------------------------
Python Front-end Example
----------------------------------------

The following code snippet shows how to construct an LTFB training
algorithm that does 10 batches of batched iterative training (e.g.,
SGD) for each local training step and uses the random pairwise
exchange strategy for metalearning. The training will perform 13
metalearning steps (i.e., "tournaments").

.. code-block:: python

   # Aliases for simplicity
   SGD = lbann.BatchedIterativeOptimizer
   RPE = lbann.RandomPairwiseExchange

   # Construct the local training algorithm
   local_sgd = SGD("local sgd", num_iterations=10)

   # Construct the metalearning strategy. This assumes the model has
   # an lbann.Metric attached to it with the name set to "accuracy".
   meta_learning = RPE(
       metric_strategies={'accuracy': RPE.MetricStrategy.HIGHER_IS_BETTER})

   # Construct the training algorithm and pass it to the trainer.
   LTFB = lbann.LTFB("ltfb",
                     local_algo=local_sgd,
                     metalearning=meta_learning,
                     metalearning_steps=13)
   trainer = lbann.Trainer(mini_batch_size=64,
                           training_algo=LTFB)



---------------------------------------------------
Truncation Selection Exchange (TSE) Variant of LTFB
---------------------------------------------------

TSE is a variant of basic LTFB that replaces random pairwise exchange (RPE)
strategy in LTFB with truncation selection exchange strategy.

In TSE, all trainers in the population set are ranked using specified
evaluation metric. Model parameters, training hyperparameters and or
topologies of any trainer in the bottom rank is replaced by that of a
(random) trainer in the top rank.

Python front end is similar to above, simply replace RPE with TSE:

.. code-block:: python

   TSE = lbann.TruncationSelectionExchange
   meta_learning = TSE(
        metric_strategies={'random': TSE.MetricStrategy.HIGHER_IS_BETTER},
        truncation_k=2)


----------------------------------------
Python Front-end API Documentation
----------------------------------------

The following is the full documentation of the Python Front-end
classes that are used to implement this execution algorithm.

.. _LTFB:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
lbann.LTFB Interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:class:: LTFB

   Livermore tournament fast-batch (LTFB) algorithm.

   This training algorithm is a simple composite training algorithm.
   The MPI universe is subdivided into several "trainers". Each
   trainer applies a local training algorithm. At the completion of
   local training, a metaheuristic ("metalearning" algorithm) is
   applied to select a new set of models to continue.

   The usage requirements for this training algorithm are a
   fully-specified local training algorithm and stopping criteria for
   the outer loop.

   .. py:class:: StoppingCriteria()

      Stopping criteria for LTFB.

      .. py:method:: __init__(metalearning_steps: int = 1)

      :param int metalearning_steps: The number of outer-loop
                                     iterations.

      .. py:method:: export_proto()

      Get a protobuf representation of this object.

      :rtype: AlgoProto.LTFB.TerminationCriteria()

   .. py:method:: __init__(name: str, local_algo: TrainingAlgorithm,
                  metalearning: MetaLearningStrategy,
                  metalearning_steps: int = 1)

      :param string name: A user-defined name to identify this object
                          in logs.

      :param TrainingAlgorithm local_algo: The trainer-local algorithm
                                           to apply.

      :param MetaLearningStrategy metalearning: The metalearning
                                                strategy to apply
                                                after local training.

      :param int metalearning_steps: The number of outer-loop
                                     iterations.

.. _MetaLearningStrategy:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
lbann.MetaLearningStrategy Interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:class:: MetaLearningStrategy()

   Base class for metalearning strategies for LTFB.

.. py:class:: MutationStrategy()

   The strategy for mutation after a tournament in LTFB.

   When a trainer loses in a LTFB tournament, the winning model is
   copied over to it and this mutation strategy is applied to the
   copied model to explore a new model. This is relevant to neural
   architecture search (NAS).

   .. py:method:: __init__(strategy: str = "null_mutation")

      :param string strategy: The LTFB metalearning strategy.

   .. py:method:: export_proto()

      Get a protobuf representation of this object.

      :rtype: AlgoProto.MutationStrategy.MutationStrategyMsg()

.. _RandomPairwiseExchange:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
lbann.RandomPairwiseExchange Interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:class:: RandomPairwiseExchange(MetaLearningStrategy)

   The classic LTFB pairwise tournament metalearning strategy.

   This metalearning strategy is the original algorithm used for
   LTFB. After each local training step, all trainers in the LBANN
   environment pair off and have a "tournament" (if the number of
   trainers is odd, one competes with itself). The tournament is a
   simple comparison of a metric value that's read from the
   model. There is an assumption that a metric with the same name
   exists in all trainers' models. The tournaments are evaluated on a
   trainer-specific data set, so it is possible that each trainer in a
   pair will think a different model has won. The winning model
   survives and is either used as the initial guess for the next round
   of local training or returned to the caller.


   Since this algorithm relies on a metric with a given name being
   present in a model, each instance of this metalearning strategy
   (and, by extension, the training algorithm in which it is used) has
   an implicit limitation on the set of models to which it can be
   applied.

   .. py:class:: MetricsStrategy()

      .. py:attribute:: LOWER_IS_BETTER: int = 0

      .. py:attribute:: HIGHER_IS_BETTER: int = 1

   .. py:class:: ExchangeStrategy

      The algorithm for exchanging model data in
      RandomPairwiseExchange.

      .. warning:: The fate of this class is under consideration. It
                   would be good for it to converge to a single
                   algorithm. Hence, no effort has been made here to
                   mirror the C++ polymorphism in this Python wrapper.

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
         (when ``exchange_hyperparameters=True``). This method
         (IMPLICITLY!!!) assumes that the weights objects to be
         exchanged appear in the same order in both the source and the
         target trainers' instances of the model. While extremely
         fragile hackery could produce other cases that happen to
         work, this essentially implies that the model topology should
         be homogenous across all trainers.

      .. py:method:: __init__(strategy: str = "checkpoint_binary",
                     weights_names: list[str] = [],
                     exchange_hyperparameters: bool = False,
                     checkpoint_dir: str = None)

         :param string strategy: Which strategy to use (default:
                                 "checkpoint_binary").

         :param list[string] weights_name: A list of weights names that
                                          should be exchanged.

         :param bool exchange_hyperparameters: If True, exchange all
                                               optimizer state. Only
                                               applies to the
                                               "sendrecv_weights"
                                               strategy.

         :param string checkpoint_dir: A path to a directory for
                                       storing the checkpoint
                                       files. Only applies to
                                       "checkpoint_file".

      .. py:method:: export_proto()

         Get a protobuf representation of this object.

         :rtype: AlgoProto.RandomPairwiseExchange.ExchangeStrategy.
                 ExchangeStrategyMsg()

         :raises: ValueError("Unknown strategy")

   .. py:method:: __init__(metric_strategies: dict[str,int] = {},
                  exchange_strategy = ExchangeStrategy(),
                  mutation_strategy = MutationStrategy())

      Construct a new RandomPairwiseExchange metalearning strategy.

      :param dict[str,int] metric_strategies: Map from metric name to
                                              the criterion for
                                              picking a winner with
                                              respect to this metric.

      :param ExchangeStrategy() exchange_strategy: The algorithm for
                                                   exchanging model
                                                   data in
                                                   RandomPairwiseExchange.

      :param MutationStrategy() mutation_strategy: The strategy for
                                                   mutation after a
                                                   tournament in
                                                   LTFB.

   .. py:method:: export_proto()

      Get a protobuf representation of this object.

      :rtype: AlgoProto.RandomPairwiseExchange()

.. py:class:: class TruncationSelectionExchange(MetaLearningStrategy)

   Truncation selection  metalearning strategy.

   Rank all trainers in a population of trainers Ranking is done using
   specified metric strategy Models/topologies/training
   hyperparameters of any trainer at ranking below truncation_k are
   replaced with that of a trainer from top of the ranking list.

   .. py:class:: MetricStrategy()

      .. py:attribute:: LOWER_IS_BETTER: int = 0

      .. py:attribute:: HIGHER_IS_BETTER: int = 1


      .. py:method:: __init__(metric_strategies: dict[str,int] = {},
                     truncation_k = 0)

         Construct a new TruncationSelectionExchange metalearning
         strategy.

         :param dict[str,int] metric_strategies: Map from metric name
                                                 to the criterion for
                                                 picking a winner with
                                                 respect to this
                                                 metric

         :param int truncation_k: Partitions ranking list to
                                  top(winners)/bottom(losers)

      .. py:method:: export_proto()

         Get a protobuf representation of this object.

         :rtype: AlgoProto.TruncationSelectionExchange()

.. py:class:: RegularizedEvolution(MetaLearningStrategy)

   This is a meta-learning strategy in population-based training. A
   sample of trainers is chosen from a population in every
   tournament. The best trainer is chosen from that sample according
   to an evaluation metric. Then the model from that best trainer is
   mutated and replaces the oldest model.

   .. py:class:: MetricStrategy()

      .. py:attribute:: LOWER_IS_BETTER: int = 0

      .. py:attribute:: HIGHER_IS_BETTER: int = 1

   .. py:method:: __init__(metric_name, metric_strategy,
                  mutation_strategy = MutationStrategy(), sample_size
                  = 0)

      :param string metric_name: The name of the metric to use for
                                 evaluation. A metric with this name
                                 must exist in the model passed to
                                 apply().

      :param string metric_strategy: Options: ``LOWER_IS_BETTER``, or
                                     ``HIGHER_IS_BETTER``.

      :param MutationStrategy() mutation_strategy: The strategy for
                                                   mutation after a
                                                   tournament in
                                                   LTFB.

      :param int sample_size: Number of trainers chosen from a
                              population in every tournament.

   .. py:method:: export_proto():

      Get a protobuf representation of this object.

      :rtype: AlgoProto.RegularizedEvolution()
