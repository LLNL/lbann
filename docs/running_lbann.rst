.. role:: bash(code)
          :language: bash

====================
Running LBANN
====================

The basic template for running LBANN is

.. code-block:: bash

    <mpi-launcher> <mpi-options> \
        lbann <lbann-options> \
        --model=model.prototext \
        --optimizer=opt.prototext \
        --reader=data_reader.prototext

When using GPGPU accelerators, users should be aware that LBANN is
optimized for the case in which one assigns one GPU per MPI
*rank*. This should be borne in mind when choosing the parameters for
the MPI launcher.

A list of options for LBANN may be found by running :bash:`lbann
--help`.

.. note:: At time of writing, it is known that some of these are
          out-of-date. An
          `issue <https://github.com/LLNL/lbann/issues/864>`_ has been
          opened to track this.

.. _using-the-model-zoo:

--------------------
Using the model zoo
--------------------

LBANN ships with prototext descriptions of a variety of models,
optimizers and data readers. These may be found in the :code:`model_zoo/`
directory of the source repository or the :code:`share/model_zoo/` directory
of the install directory.

.. warning:: Some of these prototexts point to specific data locations
             on LLNL LC clusters. Users may have to modify such paths
             to point to locations on their own systems. This can be
             done by modifying the prototext directly or overriding
             the options on the command line with, e.g., the
             :code:`--data_filedir_train` and
             :code:`--data_filedir_test` options.

The following is an example invocation of LBANN on a machine using
Slurm's :bash:`srun` as an MPI launcher. In the example command,
a machine with 2 GPGPUs per node are available, 4 nodes will be used,
:bash:`${LBANN_EXE}` is the path to the :code:`lbann` executable, and
:bash:`${LBANN_MODEL_ZOO_DIR}` is the path to the :code:`model_zoo/` directory in
either the source tree or the install tree. Note that the options
passed to :bash:`srun` are not likely to be portable to other MPI
launchers. The example will train Alexnet with SGD optimization on the
Imagenet dataset for 5 epochs.

.. code-block:: bash

    srun -N4 --ntasks-per-node=2 \
        ${LBANN_EXE} \
        --model=${LBANN_MODEL_ZOO_DIR}/models/alexnet/alexnet.prototext \
        --optimizer=${LBANN_MODEL_ZOO_DIR}/optimizers/opt_sgd.prototext \
        --reader=${LBANN_MODEL_ZOO_DIR}/data_readers/data_reader_imagenet.prototext \
        --num_epochs=5
    
---------------------------------------------
Using the Python interface for prototext
---------------------------------------------

There is a python interface for generating model prototext
files. Example Python scripts may be found in the
:code:`scripts/proto/lbann/models` directory of the source
repository. Running the Python script will generate a prototext that
can be passed to the :code:`--model` option for LBANN.

.. code-block:: bash
                
    python3 alexnet.py alexnet.prototext
    <mpi-launcher> <mpi-options> \
        lbann --model=alexnet.prototext <other-lbann-options>

where :code:`<other-lbann-options>` are as documented
:ref:`above <using-the-model-zoo>`, with optimizer and data reader
prototexts coming from the appropriate :code:`model_zoo/` directories.

------------------------------
Running the inference engine
------------------------------

This section is under construction, requiring input from other team
members. Until it is complete, please ask questions on the
`issue tracker <https://github.com/llnl/lbann/issues>`_.

