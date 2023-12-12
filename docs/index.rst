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

Users are advised to view `the Doxygen API Documentation
<_static/doxygen/html/index.html>`_ for API information.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   quick_start
   building_lbann
   running_lbann

.. toctree::
   :maxdepth: 2
   :caption: Python Front-End

   callbacks
   data_transforms
   execution_algorithms
   hyperparameter_tuning

.. toctree::
   :maxdepth: 2
   :caption: LBANN Layers

   layers
   jit_compiled_layers

.. toctree::
   :maxdepth: 2
   :caption: LBANN Operators

   operators

.. toctree::
   :maxdepth: 1
   :caption: Data Ingestion

   data_ingestion
   data_ingestion/sample_lists
   data_ingestion/hdf5_data_reader
   data_ingestion/hdf5_generate_schema_and_sample_list

.. toctree::
   :maxdepth: 1
   :caption: Publications

   publications

.. toctree::
   :maxdepth: 2
   :caption: Developer Documentation

   lbann
   lbann-api
   style_guide
   continuous_integration
   documentation_building

==================

* :ref:`genindex`
