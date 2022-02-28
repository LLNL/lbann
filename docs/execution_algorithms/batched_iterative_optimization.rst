.. role:: python(code)
          :language: python

============================================================
Batched First-Order Optimization Methods
============================================================

The most prevalent method for training neural networks is Stochastic
Gradient Descent (SGD). The basic procedure is to evaluate an
objective function with a batch of samples, compute the gradient of
the objective function at those points, and update weights using some
update rule. Over the years, as new (and quite old) problems have
revealed weaknesses in the simple SGD update step, other update rules
have emerged.

LBANN implicitly supports a variety of first-order gradient update
rules. Specifically, LBANN supports Adam, Adagrad, a hypergradient
variant of Adam, RMSprop, and classical SGD.

At this time, however, the optimization rule is part of the model
description, with a distinct rule being attached to each (nonfrozen)
weights tensor in the model. This has the advantage of flexibility:
were one so inclined, each weights tensor could update using a
different rule with a unique set of so-called
"hyperparameters". However, it strongly couples a training artifact
with the model description rather than with a training-specific
object.

The :python:`lbann.BatchedIterativeOptimizer` algorithm does not
modify this structure; neither does it expose a way to manage the
update rules or how they are applied to model parameters. Rather, it
wraps around the existing infrastructure to provide a formalism
consistent with other training algorithms in LBANN.

----------------------------------------
Python Front-end Example
----------------------------------------

Since the :python:`lbann.BatchedIterativeOptimizer` class represents a
wrapper around external infrastructure, its interface is currently
very minimal. Its behavior relies on model-level parameters (for
weights-specific update rules) and global parameters (for setting up
the default update rule).

The following example demonstrates using the SGD update rule as the
default rule. Since no other rules are set, this rule will be used for
every weights tensor in the model. The
:python:`lbann.BatchedIterativeOptimizer` is then setup to manage 10
epochs of iterative batched training with a batch size of 64. As with
all training algorithms, a name is provided to identify this training
algorithm instance in log messages.

.. code-block:: python

    # Setup elsewhere:
    model = ...
    data_reader = ...

    # Setup default optimization strategy
    default_update_rule = lbann.SGD(learn_rate=0.01, momentum=0.9)

    # Setup trainer
    trainer = lbann.Trainer(mini_batch_size=64,
                            training_algo=BatchedIterativeOptimizer(
                              "my descent algorithm",
                              epoch_count=10))

    # Pass the update rule as a global parameter to the LBANN driver
    lbann.run(trainer, model, data_reader, default_update_rule)

This is expected to be the most common use-case.

As a more advanced use-case, the following demonstrates setting a
different update rule for one weights object in the model, while using
a default update rule for the remaining weights objects. (This is
merely illustrative; there is no implication that this is a good thing
to do.)

.. code-block:: python

    # Setup a weights tensor to be optimized with Adam:
    my_weights = lbann.Weights(name="my weights",
                               optimizer=lbann.Adam(learn_rate=0.01))

    # Attach the weights tensor to a layer:
    my_layer = lbann.FullyConnected(weights=[my_weights], ...)

    # Finish setting up objects
    model = ...
    data_reader = ...

    # Setup default optimization strategy
    default_update_rule = lbann.SGD(learn_rate=0.01, momentum=0.9)

    # Setup trainer
    trainer = lbann.Trainer(mini_batch_size=64,
                            training_algo=BatchedIterativeOptimizer(
                              "my descent algorithm",
                              epoch_count=10))

    # Pass the update rule as a global parameter to the LBANN driver
    lbann.run(trainer, model, data_reader, default_update_rule)


----------------------------------------
Python Front-end API Documentation
----------------------------------------

The following is the full documentation of the Python Front-end class
that implements this training algorithm.

.. _BatchedIterativeOptimizer:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
lbann.BatchedIterativeOptimizer module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:class:: BatchedIterativeOptimizer(TrainingAlgorithm)

   Batched gradient descent training algorithm.

   Because LBANN manages optimizers on a weights-specific basis, and
   those optimizers are (theoretically) allowed to vary from weights
   to weights, it would be inaccurate to describe this as "SGD" (which
   would also clash with the "Optimizer" subclass). Nonetheless, this
   is the well-loved first-order descent algorithm that seems good
   enough most of the time for most people (or so seems the general
   consensus).

   Also as a result of the way in which LBANN manages optimizers, the
   usage requirements for this class are quite sparse: only the
   stopping criteria are specified. The rest of the behavior of this
   class is specified in the model.

   .. py:class:: StoppingCriteria()

      Stopping criteria for an instance of BatchedIterativeOptimizer.

      BatchedIterativeOptimizer can be done to a finite batch count, a
      finite epoch count, or a fixed number of (double-precision)
      seconds (note: time-based stopping criteria is not publicly
      available yet; this is just a bit of foreshadowing).

      When used as a subalgorithm within a composite algorithm (see
      LTFB, below), this specifies the stopping criteria for each
      invocation. The algorithm stops when any of the criteria is
      satisfied.

      .. py:method:: __init__(batch_count: int = 0, epoch_count: int =
                     0, seconds: float = 0.)

         Construct a new BatchedIterativeOptimizer stopping criteria.

         :param int batch_count: Number of minibatches.

         :param int epoch_count: Number of epochs.

         :param float seconds: Number of epochs.

      .. py:method:: export_proto()

         Get a protobuf representation of this object.

         :rtype: AlgoProto.TrainingAlgorithm()

   .. py:method:: __init__(name: str, num_iterations: int = 0,
                  epoch_count: int = 0, max_seconds: float = 0.)

      Construct a new BatchedIterativeOptimizer instance.

      :param string name: A user-defined name to identify this object
                          in logs.

      :param int num_iterations: Number of minibatches.

      :param int epoch_count: Number of epochs.

      :param float max_seconds: Maximum training duration (seconds).

   .. py:method:: do_export_proto()

      Get a protobuf representation of this object.

      :rtype: AlgoProto.SGD()
