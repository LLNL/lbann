.. role:: bash(code)
          :language: bash

====================
Quick Start
====================

--------------------
What can LBANN do?
--------------------

The Livermore Big Artificial Neural Network toolkit (LBANN) is an HPC-centric
deep learning training framework that works across multiple levels of
parallelism.  LBANN is capable of taking advantage of HPC hardware to
accelerate the training of deep learning models on massive datasets.


--------------------
Installing LBANN
--------------------

LBANN supports installation through Spack and CMake.  We recommend using the
Spack installation instructions below.  If the Spack install fails, try using
the :ref:`CMake install <build-with-cmake>`.

1.  Download and install `Spack <https://github.com/llnl/spack>`_.  Enable the
    additional Spack commands for module files described `here
    <https://spack.readthedocs.io/en/latest/module_file_support.html#id2>`_:

    .. code-block:: bash

        source ${SPACK_ROOT}/share/spack/setup-env.sh

2.  Users that are `familiar with Spack
    <https://spack-tutorial.readthedocs.io/en/latest/tutorial_basics.html>`_
    and already have a `custom Spack ecosystem
    <https://spack.readthedocs.io/en/latest/configuration.html>`_ can install
    LBANN with:

    .. code-block:: bash

        spack install lbann <customization options>

    A complete list of LBANN install options can be found with:

    .. code-block:: bash

        spack info lbann

    For users new to Spack, LBANN provides a script that will perform some
    basic configuration (e.g., add paths to externally installed packages) and
    install LBANN in a Spack environment.  *This script is only tested and
    maintained for systems at LLNL, NERSC, and ORNL.  If you are not running on
    a system at one of these institutions, you may try the Spack install above
    or the :ref:`CMake install <build-with-cmake>`.* To use this installation
    script, clone the repository and run the script:

    .. code-block:: bash

        git clone https://github.com/llnl/lbann
        cd ./lbann
        ./scripts/build_lbann.sh -d -- +cuda +half

    View other options available by passing the :code:`-h` option to the
    script.

.. note:: It is recommended that your Spack environment take advantage
          of locally installed tools.  Unless your Spack environment
          is explicitly told about tools such as CMake, Python, MPI,
          etc., it will install everything that LBANN and all of its
          dependencies require. This can take quite a long time but
          only has to be done once for a given spack repository. Once
          all of the standard tools are installed, rebuilding LBANN
          with Spack is quite fast.

          Advice on setting up paths to external installations is
          beyond the scope of this document but is covered in the
          `Spack Documentation
          <https://spack.readthedocs.io/en/latest/configuration.html>`_.


.. _test-lbann-install:

--------------------
Test LBANN Install
--------------------

0. [HPC Center Option] If you are on typical HPC system you may want
   to get an allocation on a compute node to compile your code and run
   your tests.  See your compute center's policy and documentation for
   where you should build and run code.  Note that the LBANN Python
   Front End (PFE) launcher can run from an allocated compute node or
   dispatch to a set of known job launchers:

    .. code-block:: bash

        <get some sort of compute node>

1. If you used the :code:`build_lbann.sh` script for installation or
   installed in a Spack environment, you will need to activate the Spack LBANN
   environment:

    .. code-block:: bash

        spack env activate -p lbann

2. Test an implementation of the `LeNet neural network
   <http://yann.lecun.com/exdb/lenet/>`_ on the `MNIST data set
   <https://en.wikipedia.org/wiki/MNIST_database>`_ at :code:`<lbann repo
   path>/applications/vision/lenet.py` to verify that your LBANN installation
   is working correctly:

    .. code-block:: bash

        cd <lbann repo path>/applications/vision/
        python3 lenet.py

    Running this Python script will automatically submit a job to the system
    scheduler.  If LBANN was built successfully, you should see output from
    LBANN about loading the data, building the network, and training the model.

    If LBANN fails to run, you can view the generated job script and log files,
    and run the job manually:

    .. code-block:: bash

        ls ./\*_lbann_lenet

    If this also fails, you may try building LBANN again using the :ref:`CMake
    install instructions <build-with-cmake>`.


--------------------
Basic Usage
--------------------

A typical workflow involves the following steps:

1. Configuring a :python:`Trainer`.

2. Configuring LBANN model components (like the graph of
   :python:`Layer` s) and creating a :python:`Model`.

  + Classes for model components are automatically generated from the
    LBANN Protobuf specifications in `lbann/src/proto
    <https://github.com/LLNL/lbann/blob/develop/src/proto>`_. These
    files are currently the best source of documentation. Message
    fields in the Protobuf specification are optional keyword
    arguments for the corresponding Python class constructor. If a
    keyword argument is not provided, it is logically zero (e.g. false
    for Boolean fields and empty for string fields)

3. Configuring the default :python:`Optimizer` to be used by the
   :python:`Weights` objects.

4. Loading in a Protobuf text file describing the data reader.

   + The Python frontend currently does not have good support for
     specifying data readers. If any data reader properties need to be
     set programmatically, the user must do it directly via the
     Protobuf Python API.

5. Launching LBANN by calling :python:`run`.

   + :python:`lbann.run` should be run from a compute node. If a node
     allocation is not available, the :python:`batch_job` option can
     be set to submit a batch job to the scheduler.

   + A timestamped work directory will be created each time LBANN is
     run. The default location of these work directories can be set
     with the environment variable :bash:`LBANN_EXPERIMENT_DIR`.

   + Supported job managers are Slurm and LSF.

   + LLNL users and collaborators may prefer to use
     :python:`lbann.contrib.launcher.run`. This is similar to
     :python:`lbann.run`, with defaults and optimizations for certain
     systems.


--------------------
PyTorch to LBANN
--------------------

The LBANN Python API is very similar to the PyTorch API.  In order to help
users familiar with PyTorch transition to LBANN, we prepared the following
guide:

~~~~~~~~~~~~~~~~~~~~
Loading Data
~~~~~~~~~~~~~~~~~~~~
Both LBANN and PyTorch use similar strategies for loading data into models.
With PyTorch, we can load the `MNIST dataset
<https://en.wikipedia.org/wiki/MNIST_database>`_ using the included
:python:`DataLoader`:

    .. code-block:: python

        import torch
        from torchvision import datasets, transforms

        batch_size = 64
        data_loader = torch.utils.data.DataLoader(
                      datasets.MNIST('data', train=True, download=True,
                                     transform=transforms.ToTensor()),
                      batch_size=batch_size)

With LBANN, you can write custom data reader functions that use protobuf files
to define the input data and transform it into the input tensors for your
model:

    .. code-block:: python

        import os
        import lbann
        from google.protobuf import text_format

        def make_data_reader(data_dir):
            protobuf_file = os.path.join(data_dir, 'data_reader.prototext')
            message = lbann.lbann_pb2.LbannPB()
            with open(protobuf_file, 'r') as f:
                text_format.Merge(f.read(), message)
            message = message.data_reader
            message.reader[0].data_filedir = data_dir

            return message

        data_reader = make_data_reader(os.path.realpath('./mnist_data/'))

This reader assumes that the files `train-images-idx3-ubyte
<https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz>`_,
`train-labels-idx1-ubyte
<https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz>`_, and
:code:`data_reader.prototext` are located in the :bash:`./mnist_data`
directory.  The :code:`data_read.prototext` file contains the following:

    .. code-block:: protobuf

        data_reader {
          reader {
            name: "mnist"
            role: "train"
            shuffle: true
            data_filedir: "mnist_data"
            data_filename: "train-images-idx3-ubyte"
            label_filename: "train-labels-idx1-ubyte"
            validation_fraction: 0.1
            fraction_of_data_to_use: 1.0
            transforms {
              scale {
                scale: 0.003921568627  # 1/255
              }
            }
          }
        }

~~~~~~~~~~~~~~~~~~~~
Building a Model
~~~~~~~~~~~~~~~~~~~~

Building models in LBANN is similar to building models in PyTorch.
For example, we can define a simple PyTorch model for the MNIST dataset with:

    .. code-block:: python

        import torch.nn as nn
        import torch.nn.functional as F

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv = nn.Conv2d(1, 20, kernel_size=5)
                self.fc = nn.Linear(12*12*20, 10)

            def forward(self, x):
                x = self.conv(x)
                x = F.relu(x)
                x = F.max_pool2d(x, 2)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                x = F.log_softmax(x, dim=1)
                return x

        net = Net()


Using LBANN, that same neural network can be built with:

    .. code-block:: python

        input_ = lbann.Input(target_mode = 'classification')
        images = lbann.Identity(input_)
        labels = lbann.Identity(input_)

        x = lbann.Convolution(images, num_dims=2, out_channels=20,
                              num_groups=1, kernel_size=5, stride=1,
                              dilation=1, has_bias=True)
        x = lbann.Relu(x)
        x = lbann.Pooling(x, num_dims=2, pool_dims_i=2,
                          pool_strides_i=2, pool_mode='max')
        x = lbann.FullyConnected(x, num_neurons=10, has_bias=True)
        probs = lbann.Softmax(x)

        loss = lbann.CrossEntropy(probs, labels)

        model = lbann.Model(epochs=5,
                            layers=lbann.traverse_layer_graph(input_),
                            objective_function=loss,
                            callbacks=[lbann.CallbackPrintModelDescription(),
                                       lbann.CallbackPrint()])

~~~~~~~~~~~~~~~~~~~~
Setup Model Training
~~~~~~~~~~~~~~~~~~~~

Training a model with PyTorch can be achieved by setting a few parameters,
defining an optimizer, and building a training loop:

    .. code-block:: python

        import torch.optim as optim

        learning_rate = 0.01
        momentum = 0.5

        opt = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

        def train(epoch):
            net.train()
            for batch_idx, (data, target) in enumerate(data_loader):
                opt.zero_grad()
                output = net(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                opt.step()

            print('Training Epoch: {},\tLoss: {:.3f}'.format(epoch, loss.item()))

With LBANN, we also define learning parameters and an optimizer.  With LBANN,
a :python:`Trainer` is provided that negates the need to build your own
training loop:

    .. code-block:: python

        learning_rate = 0.01
        momentum = 0.5
        batch_size = 64

        opt = lbann.SGD(learn_rate=learning_rate, momentum=momentum)

        trainer = lbann.Trainer(mini_batch_size=batch_size)

~~~~~~~~~~~~~~~~~~~~
Run the Experiment
~~~~~~~~~~~~~~~~~~~~

Running the experiment in PyTorch is as simple as calling the training loop:

    .. code-block:: python

        for epoch in range(5):
            train(epoch)

Running the experiment in LBANN is just as easy:

    .. code-block:: python

        import lbann.contrib.launcher
        lbann.contrib.launcher.run(trainer, model, data_reader,
                                   opt, job_name='mnist-test')

Python acts only as a frontend for LBANN.  The above commands will
automatically generate a batch job script and submit it to the system
scheduler.  You can see the job script and associated job files in the
:bash:`./*mnist-test/` directory.

.. note:: The LBANN :python:`launcher.run` can accept additional arguments to
          specify additional scheduler and job parameters.  LBANN provides
          methods that help with these parameters at
          :python:`lbann.contrib.args.add_scheduler_arguments()` and
          :python:`lbann.contrib.args.get_scheduler_kwargs()`.
