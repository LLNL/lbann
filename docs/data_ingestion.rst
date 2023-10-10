Data Ingestion
==============

Getting data into LBANN requires using one of the predefined data
readers or writing a customized data reader tailored to the data in
question. Currently, data readers serve two roles:

1. Define how to ingest data from storage (at rest) and place it into
   an LBANN-compatible format

2. Understand the structure of a well-defined ("named") data set such
   as MNIST or ImageNet-1K (ILSVRC).

As LBANN is evolving, we are working to separate these two behaviors
into distinct objects, but it is still a work in progress.  As a
result there are some "legacy" data readers that represent both of
these features, and some new data readers that focus more on task 1
and incorporate the use of a sample list to help with task 2.

At this time, LBANN can only ingest static data sets; work on
streaming data is in progress.


Legacy Data Readers
-------------------

Some of the legacy data readers are the ``MNIST``, ``ImageNet``, and
``CIFAR10`` data readers.


"New" Data Readers
-------------------

Two of the new format data readers are the ``python``, ``SMILES``, and
:ref:`HDF5<sec:hdf5_data_reader>` readers.

Several of these readers (SMILES and
:ref:`HDF5<sec:hdf5_data_reader>`) support the use of :ref:`sample
lists<sec:sample-lists>`.

Iterative algorithms and data ingestion
---------------------------------------
One of the challenges of data ingestion is managing the interplay
between the size of the data set and the size of the mini-batch
consumed by each step of the execution algorithm.  With respect to the
model and execution algorithm there are two key fields that capture
this information:

1) the maximum mini-batch size (`max_mini_batch_size`) that a model is
   configured to support.  Nominally this is dictates how much memory
   is allocated in each tensor, and is established and then allocated
   during the model setup.  It is a property of the model.  Note that
   at the current moment, if the current mini-batch size exceeds the
   maximum mini-batch size, a warning is thrown and then the matrices
   are resized.

2) the current mini-batch size (`current_mini_batch_size`), which is
   dictated by how much data is available from the data ingestion
   pipeline.  This value can vary from step to step, but typically is
   equal to the maximum mini-batch size for all but the last step of
   an execution algorithm (when the data ingestion pipeline has run
   out of data).  The field for the current mini-batch size is
   governed by the data readers and then is cached in both the model
   as well as the current execution context.  Note that it is not
   clear if the execution contexts should hold this data anymore.
