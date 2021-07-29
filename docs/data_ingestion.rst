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
