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
trainers in the unverse are paired off, with each set of partners
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

Python front end is similar to above. simply replace RPE with TSE:

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

.. autoclass:: lbann.LTFB
   :members:
   :undoc-members:
   :special-members: __init__

.. autoclass:: lbann.MetaLearningStrategy
   :members:
   :undoc-members:
   :special-members: __init__

.. autoclass:: lbann.RandomPairwiseExchange
   :members:
   :undoc-members:
   :special-members: __init__
