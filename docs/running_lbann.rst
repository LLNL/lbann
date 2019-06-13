.. role:: bash(code)
          :language: bash
.. role:: python(code)
          :language: python

============================================================
Running LBANN
============================================================

------------------------------------------------
Anatomy of an LBANN experiment
------------------------------------------------

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Parallelism
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

LBANN is run under `MPI
<https://en.wikipedia.org/wiki/Message_Passing_Interface>`_, i.e. with
multiple processes that communicate with message passing. This set of
processes is subdivided into one or more "trainers." Conceptually, a
trainer owns parallel objects, like models and data readers, and
generally operates independently of other trainers.

Comments:

+ LBANN targets HPC systems with homogeneous compute nodes and GPU
  accelerators, which motivates some simplifying assumptions:

  - All trainers have the same number of processes.

  - If GPU acceleration is enabled, each MPI process corresponds to
    one GPU.

+ Processors are block assigned to trainers based on MPI rank.

  - In order to minimize the cost of intra-trainer communication, make
    sure to map processes to the hardware and network
    topologies. Typically, this just means choosing a sensible number
    of processes per trainer, e.g. a multiple of the number of GPUs
    per compute node.

+ Generally, increasing the number of processes per trainer will
  accelerate computation but require more intra-trainer
  communication. There is typically a sweet spot where run time is
  minimized, but it is complicated and sensitive to the nature of the
  computation, the mini-batch size, the data partitioning scheme,
  hardware and network properties, the communication algorithms, and
  myriad other factors.

  - Rule-of-thumb: Configure experiments so that the bulk of run time
    is taken by compute-bound operations (e.g. convolution or matrix
    multiplication) and so that each process has enough work to
    achieve a large fraction of peak performance (e.g. by making the
    mini-batch size sufficiently large).

+ Most HPC systems are managed with job schedulers like `Slurm
  <https://slurm.schedmd.com/overview.html>`_. Typically, users can
  not immediately access compute nodes but must request them from
  login nodes. The login nodes can be accessed directly (e.g. via
  :bash:`ssh`), but users are discouraged from doing heavy computation
  on them.

  - For debugging and quick testing, it's convenient to request an
    interactive session (:bash:`salloc` or :bash:`sxterm` with Slurm).

  - If you need to run multiple experiments or if experiments are not
    time-sensitive, it's best to submit a batch job (:bash:`sbatch`
    with Slurm).

  - When running an experiment, make sure you know what scheduler
    account to charge (used by the scheduler for billing and
    determining priority) and what scheduler partition to run on
    (compute nodes on a system are typically subdivided into multiple
    groups, e.g. for batch jobs and for debugging).

    + With :bash:`salloc`, specify the partition using the
      :bash:`--partition` command-line argument and specify the
      account using :bash:`--account`.

  - Familiarize yourself with the rules for the systems you use
    (e.g. the expected work for each partition, time limits, job
    submission limits) and be a good neighbor.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Model components
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. warning:: `A major refactor of core model infrastructure
             <https://github.com/LLNL/lbann/pull/916>`_ is
             pending. This documentation will be updated once it is
             merged and the interface stabilized.

+ Layer: A tensor operation, arranged within a directed acyclic graph.

  - During evaluation ("forward prop"), a layer receives input tensors
    from its parents and sends an output tensor to each child.

  - During automatic differentiation ("backprop"), a layer receives
    "input error signals" (objective function gradients w.r.t. output
    tensors) from its children and sends "output error signals"
    (objective function gradients w.r.t. input tensors) to its
    parents. If the layer has any associated weight tensors, it will
    also compute objective function gradients w.r.t. the weight
    tensors.

  - Most layers require a specific number of parents and children, but
    LBANN will insert layers into the graph if there is a mismatch and
    the intention is obvious. For example, if a layer expects one
    child but has multiple, then a split layer (with multiple output
    tensors all identical to the input tensor) is inserted. Similarly,
    if a layer has fewer children than expected, dummy layers will be
    inserted. However, this does not work if there is any
    ambiguity. In such cases (common with input and slice layers), it
    is recommended to manually insert identity layers so that the
    parent/child relationships are absolutely unambiguous.

+ Weights: A tensor consisting of trainable parameters, typically
  associated with one or more layers. A weight tensor owns an
  initializer to initially populate its values and an optimizer to
  find values that minimize the objective function.

  - A weight tensor without a specified initializer will use a zero
    initializer.

  - A weight tensor without a specified optimizer will use the model's
    default optimizer.

  - If a layer requires weight tensors and none are specified, it will
    create the needed weight tensors. The layer will pick sensible
    initializers and optimizers for the weight tensors. For example, a
    convolution layer will initialize its kernel tensor with He normal
    initialization and with the model's default optimizer.

  - The dimensions of a weight tensor is determined by their
    associated layers. The user can not set it directly.

+ Objective function: Mathematical expression that the optimizers will
  attempt to minimize. It is made up of multiple terms that are added
  together (possibly with scaling factors).

  - An objective function term can get its value from a scalar-valued
    layer, i.e. a layer with an output tensor with one entry.

+ Metric: Mathematical expression that will be reported to the
  user. This typically does not affect training, but is helpful for
  evaluating the progress of training. A canonical example for
  classification problems is classification accuracy.

+ Callback: Function that is performed at various points during an
  experiment. Callbacks are helpful for reporting, debugging, and
  performing advanced training techniques.

  - This is the natural home for experimental training
    techniques.

  - A common use-case is to export values with the "dump outputs"
    callback so that the user can perform data post-processing or
    visualization.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Data readers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. warning:: The core infrastructure for data readers is slated for
             significant refactoring, so expect major changes in the
             future.

Data readers are responsible for managing a data set and providing
data samples to models. A data set is comprised of independent data
samples, each of which is made up of multiple tensors. For example, a
data sample for a labeled image classification problem consists of an
image tensor and a one-hot label vector.

.. note:: The data readers are currently hard-coded to assume this
          simple classification paradigm. Hacks are needed if your
          data does not match it exactly, e.g. if a data sample is
          comprised of more than two tensors. The most basic approach
          is to flatten all tensors and concatenate them into one
          large vector. The model is then responsible for slicing this
          vector into the appropriate chunks and resizing the chunks
          into the appropriate dimensions. Done correctly, this should
          not impose any additional overhead.

Specifically, data readers and models interact via input layers. Each
model must have exactly one input layer and its output tensors are
populated by a data reader every mini-batch step. This is typically
performed by a background thread pool, so data ingestion will
efficiently overlap with other computation, especially if the data
reader's work is IO-bound or if the computation is largely on GPUs.

.. note:: An input layer has an output tensor for each data sample
          tensor. Since each data sample has two tensors (one for the
          data and one for the label), it follows that every input
          layer should have two child layers. To make parent/child
          relationships unambiguous, we recommend manually creating
          identity layers as children of the input layer.

Note that layers within a model treat the data for a mini-batch as a
single tensor where the leading dimension is the mini-batch
size. Thus, corresponding tensors in all data samples must have the
same dimensions. The data dimensions must be known from the beginning
of the experiment and can not change. However, real data is rarely so
consistent and some preprocessing is typically required.

.. warning:: `A major refactor of the preprocessing pipeline
             <https://github.com/LLNL/lbann/pull/1014>`_ is
             pending. This documentation will be updated once it is
             merged and the interface stabilized.

------------------------------------------------
Python frontend
------------------------------------------------

LBANN provides a Python frontend with syntax reminiscent of `PyTorch
<https://pytorch.org/>`_. See the `model zoo implementation of LeNet
<https://github.com/LLNL/lbann/blob/develop/model_zoo/vision/lenet.py>`_
for a simple example.

Comments:

+ Under-the-hood, the Python frontend is actually a convenience
  wrapper around the Protobuf frontend. The core infrastructure allows
  users to configure an experiment and "compiles" it to a Prototext
  text file.

+ The Python interface can only configure and launch experiments. It
  is not active during an experiment and it does not allow for any
  dynamic control flow.

+ Only Python 3 is supported.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :python:`lbann` Python package is installed as part of the LBANN
build process. However, it is necessary to update the
:bash:`PYTHONPATH` environment variable to make sure Python detect
it. There are several ways to do this:

+ If LBANN has been built with the Spack user build process, loading
  LBANN will automatically update :bash:`PYTHONPATH`:

.. code-block:: bash

    module load lbann

.. warning:: The above will *not* work if LBANN has been built with
             :bash:`scripts/build_lbann_lc.sh` or with the Spack
             developer build process.

+ LBANN includes a modulefile that updates :bash:`PYTHONPATH`:

.. code-block:: bash

    module use <install directory>/etc/modulefiles
    module load lbann-<version>

+ Directly manipulate :bash:`PYTHONPATH`:

.. code-block:: bash

    export PYTHONPATH=<install directory>/lib/python<version>/site-packages:${PYTHONPATH}

Note that LBANN depends on the Protobuf Python package, which can be
installed with:

.. code-block:: bash

    pip install protobuf

If the user does not own the site-packages directory, then it may be
necessary to pass the :bash:`--user` flag to pip.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Basic usage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A typical workflow involves the following steps:

1. Configuring LBANN model components (like the graph of
   :python:`Layer` s) and creating a :python:`Model`.

  + Classes for model components are automatically generated from the
    LBANN Protobuf specification at `src/proto/lbann.proto
    <https://github.com/LLNL/lbann/blob/develop/src/proto/lbann.proto>`_.
    This file is currently the best source of documentation. Message
    fields in the Protobuf specification are optional keyword
    arguments for the corresponding Python class constructor.

2. Configuring the default :python:`Optimizer` to be used by the
   :python:`Weights` es.

3. Loading in a Protobuf text file describing the data reader.

   + The Python frontend currently does not have good support for
     specifying data readers. If any data reader properties need to be
     set programmatically, the user must do it directly via the
     Protobuf Python API.

4. Launching LBANN by calling :python:`run`.

   + :python:`lbann.run` will detect whether the user is currently on
     a login node or a compute node. If on a login node, a batch job
     will be submitted to the job scheduler. If on a compute node,
     LBANN will be run directly on the allocated nodes.

   + A timestamped work directory will be created each time LBANN is
     run. The default location of these work directories can be set
     with the environment variable :bash:`LBANN_EXPERIMENT_DIR`.

   + Supported job managers are Slurm and LSF.

   + LLNL users may prefer to use :python:`lbann.contrib.lc.launcher.run`.
     This is a wrapper around :python:`lbann.run`, with defaults and
     optimizations specifically for LC systems.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A simple example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import lbann

    # ----------------------------------
    # Construct layer graph
    # ----------------------------------

    # Input data
    input = lbann.Input()
    image = lbann.Identity(input)
    label = lbann.Identity(input)

    # Softmax classifier
    y = lbann.FullyConnected(image, num_neurons = 10, has_bias = True)
    pred = lbann.Softmax(y)

    # Loss function and accuracy
    loss = lbann.CrossEntropy([pred, label])
    acc = lbann.CrossEntropy([pred, label])

    # ----------------------------------
    # Setup experiment
    # ----------------------------------

    # Setup model
    mini_batch_size = 64
    num_epochs = 5
    model = lbann.Model(mini_batch_size,
                        num_epochs,
                        layers=lbann.traverse_layer_graph(input),
                        objective_function=loss,
                        metrics=[lbann.Metric(acc, name='accuracy', unit='%')],
                        callbacks=[lbann.CallbackPrint(), lbann.CallbackTimer()])

    # Setup optimizer
    opt = lbann.SGD(learn_rate=0.01, momentum=0.9)

    # Load data reader from prototext
    import google.protobuf.text_format as txtf
    data_reader_proto = lbann.lbann_pb2.LbannPB()
    with open('path/to/lbann/model_zoo/data_readers/data_reader.prototext', 'r') as f:
        txtf.Merge(f.read(), data_reader_proto)
    data_reader_proto = data_reader_proto.data_reader

    # ----------------------------------
    # Run experiment
    # ----------------------------------

    lbann.run(model, data_reader_proto, opt)

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Useful submodules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

^^^^^^^^^^^^^^^^^^^^^^^^
:python:`lbann.modules`
^^^^^^^^^^^^^^^^^^^^^^^^

A :python:`Module` is a pattern of layers that can be applied multiple
times in a neural network. Once created, a :python:`Module` is
*callable*, taking a layer as input and returning a layer as
output. They will create and manage :python:`Weights` es internally,
so they are convenient for weight sharing between different
layers. They are also useful for complicated patterns like RNN cells.

*A possible note of confusion*: "Modules" in LBANN are similar to
"layers" in PyTorch, TensorFlow, and Keras. LBANN uses "layer" to
refer to tensor operations, in a similar manner as Caffe.

^^^^^^^^^^^^^^^^^^^^^^^^
:python:`lbann.models`
^^^^^^^^^^^^^^^^^^^^^^^^

Several common and influential neural network models are implemented
as :python:`Module` s. They can be used as building blocks within more
complicated models.

^^^^^^^^^^^^^^^^^^^^^^^^
:python:`lbann.proto`
^^^^^^^^^^^^^^^^^^^^^^^^

The :python:`save_prototext` function will export a Protobuf text
file, which can be fed into the Protobuf frontend.

^^^^^^^^^^^^^^^^^^^^^^^^
:python:`lbann.onnx`
^^^^^^^^^^^^^^^^^^^^^^^^

This contains functionality to convert between LBANN and ONNX
models. See `python/docs/onnx/README.md
<https://github.com/LLNL/lbann/blob/develop/python/docs/onnx/README.md>`_
for full documentation.

------------------------------------------------
Protobuf frontend (advanced)
------------------------------------------------

The main LBANN driver uses Protobuf text files (sometimes called
prototext files) to specify experiments. The Python frontend operates
by "compiling" an experiment configuration into a Protobuf text file
and passing it into the LBANN driver. Aside from quick debugging,
there are very few situations where directly manipulating Protobuf
text files is superior to using the Python frontend. In fact, it is
possible to use Protobuf's Python API to programmatically manipulate
Protobuf messages, if such fine control is necessary.

In order to fully specify an experiment, the user must provide
Protobuf text files for the model, default optimizer, and data
reader. These can be provided as three separate files or one unified
file. The basic template for running LBANN is

.. code-block:: bash

    <mpi-launcher> <mpi-options> \
        lbann --prototext=experiment.prototext

The LBANN Protobuf format is defined in `src/proto/lbann.proto
<https://github.com/LLNL/lbann/blob/develop/src/proto/lbann.proto>`_. It
is important to remember that the default value of a Protobuf field is
logically zero (e.g. false for Boolean fields and empty for string
fields).
