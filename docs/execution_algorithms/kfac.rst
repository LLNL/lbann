.. role:: python(code)
          :language: python

============================================================
Kronecker-Factored Approximate Curvature (KFAC)
============================================================

The Kronecker-Factored Approximate Curvature (KFAC) algorithm is an
approach that uses a second-order optimization method. It shows similar
capabilities to first-order methods, but converges faster and can handle
larger mini-batches. See `Large-Scale Distributed Second-Order Optimization Using Kronecker-Factored Approximate Curvature for Deep Convolutional Neural Networks <https://arxiv.org/pdf/1811.12019.pdf>`_.

----------------------------------------
Python Front-end API Documentation
----------------------------------------

.. py:class:: KFAC(TrainingAlgorithm)

   Kronecker-Factored Approximate Curvature algorithm.

   Applies second-order information to improve the quality of
   gradients in SGD-like optimizers.

   .. py:method:: __init__(name: str, first_order_optimizer:
                  BatchedIterativeOptimizer, **kfac_args)

      Construct a new KFAC algorithm.

      :param string name: A user-defined name to identify this object
                          in logs.

      :param BatchedIterativeOptimizer first_order_optimizer:  The
                                                               SGD-like
                                                               algorithm
                                                               to
                                                               apply.

      :param \**kfac_args: See `KFAC message
                           <https://github.com/LLNL/lbann/blob/fbc9d5cfe924e25c383ba3f1fb24fadab98e76a0/src/proto/training_algorithm.proto#L152>`_
                           for a list of kwargs.

   .. py:method:: do_export_proto()

      Get a protobuf representation of this object.

      :rtype: AlgoProto.KFAC()
