.. role:: python(code)
          :language: python
.. role:: cpp(code)
          :language: c++

.. _execution_algorithms:

============================================================
Execution Algorithms
============================================================

LBANN's drivers support several different execution algorithms. In
particular, LBANN supports a basic (batched) inference algorithm as
well as a variety of algorithms for training neural networks. The
execution algorithms are implemented in C++, and their parameters (or
"hyperparameters") are exposed to users via the Python Front-End
(PFE).

-------------------------
Batched Inference
-------------------------

This algorithm is not yet documented.

-------------------------
Training Algorithms
-------------------------

A training algorithm (C++: :cpp:`lbann::training_algorithm`, Python:
:python:`lbann.TrainingAlgorithm`) is the method for optimizing a
model's trainable parameters (weights). At the C++ level, a training
algorithm takes as input an initial model description (future: a
collection of model descriptions), a data source, and some stopping
criteria. Once a training algorithm has reached its prescribed
stopping criteria, it is defined to be "trained" and the updated model
description (future: collection of model descriptions) is returned to
the user.

At the PFE level, the model (future: models) and data source are not
yet properly associated with the training algorithm; fixing this issue
is work in progress. Instead, the training algorithm is associated
with the trainer object. The model (future: models) and data source
components are managed separately (C++: :cpp:`lbann::model` and
:cpp:`lbann::data_coordinator`; Python: :python:`lbann.Model` and
:python:`lbann.DataReader`, respectively) and properly associated with
the trainer's training algorithm object in the C++ runtime.

An example description of a training algorithm is shown below.

.. code-block:: python

   SGD = lbann.BatchedIterativeOptimizer   # Just an alias
   trainer = lbann.Trainer(training_algo=SGD("my sgd", epoch_count=20))

The first positional argument to every training algorithm is a
user-defined name. This will be useful for identifying this algorithm
in log messages, especially in the case of complex composite
algorithms that might use multiple or nested instances of the same
algorithm. Remaining (keyword) arguments are generally
algorithm-dependent and users should consult the :python:`help()`
messages or the API documentation for the specific algorithms they
wish to use.

----------------------------------------
Python Front-end API Documentation
----------------------------------------

.. _TrainingAlgorithm:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
lbann.TrainingAlgorithm module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:class:: TrainingAlgorithm()

The :python:`lbann.TrainingAlgorithm` is the base class of all
training algorithms used in the Python Front-end.

   .. py:method:: __init__(name: str)

      Construct a training algorithm.

      :param string name: A user-defined name to identify this
                          object in logs.

   .. py:method:: export_proto()

      Get a protobuf representation of this object.

      :rtype: AlgoProto.TrainingAlgorithm()

   .. py:method:: do_export_proto()

      Get a protobuf representation of this object.

      .. important:: Must be implemented in derived classes.

      :raises: NotImplementedError

.. _available-exe_algos:

------------------------------------------------
Supported algorithms
------------------------------------------------

.. toctree::
   :maxdepth: 1

   Batched inference <execution_algorithms/batched_inference>
   Batched first-order optimization <execution_algorithms/batched_iterative_optimization>
   LTFB <execution_algorithms/ltfb>
   KFAC <execution_algorithms/kfac>
