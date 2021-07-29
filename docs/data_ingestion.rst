Data Ingestion
==============

Getting data into LBANN requires using one of the pre-defined data
readers, or writing your own.  Currently, data readers serve two
roles: 1) define how to ingest data from storage (at rest) and place
it into an LBANN compatible format, and 2) understand the structure of
a well defined ("named") data set such as MNIST or ImageNet-1K
(ILSVRC).  As LBANN is evolving, we are working to separate these two
behaviors into distinct objects, but it is still a work in progress.
As a result there are some "legacy" data readers that represent both
of these features, and some new data readers that focuse more on task
1 and incorporate the use of a sample list to help with task 2.

Legacy Data Readers
-------------------

Some of the current legacy data readers are: ``MNIST``, ``ImageNet``, ``CIFAR10``

"New" Data Readers
-------------------

Two of the new format data readers are the ``python``, ``SMILES``, and :ref:`HDF5<sec:hdf5_data_reader>`
readers.

Several of these readers (SMILES and :ref:`HDF5<sec:hdf5_data_reader>`) support the use of :ref:`sample
lists<sec:sample-lists>`.
