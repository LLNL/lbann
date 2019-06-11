.. lbann documentation master file

LBANN: Livermore Big Artificial Neural Network Toolkit
============================================================

The Livermore Big Artificial Neural Network toolkit (LBANN) is an
open-source, HPC-centric, deep learning training framework that is
optimized to compose multiple levels of parallelism.

LBANN provides model-parallel acceleration through domain
decomposition to optimize for strong scaling of network training.  It
also allows for composition of model-parallelism with both data
parallelism and ensemble training methods for training large neural
networks with massive amounts of data.  LBANN is able to take advantage of
tightly-coupled accelerators, low-latency high-bandwidth networking,
and high-bandwidth parallel file systems.

LBANN supports state-of-the-art training algorithms such as
unsupervised, self-supervised, and adversarial (GAN) training methods
in addition to traditional supervised learning.  It also supports
recurrent neural networks via back propagation through time (BPTT)
training, transfer learning, and multi-model and ensemble training
methods.


.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   building_lbann
   running_lbann

.. toctree::
   :maxdepth: 1
   :caption: Publications

   publications

.. toctree::
   :maxdepth: 2
   :caption: Developer Documentation

   lbann/lbann
   style_guide
   continuous_integration
   documentation_building

==================

* :ref:`genindex`
