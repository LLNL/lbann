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

"Really New" Data Subsystem
---------------------------

During execution LBANN will ingest one or more streams of data.  There
will be unique streams of data for each execution mode:
 - training
 - validation
 - tournament
 - testing
 - inference

Note that execution modes should become more flexible and should be
able to be arbitrarily named.

The data stream object is responsible for keeping track of the "count"
/ state of that data stream for that execution context.  For bounded /
batched data streams, this would be the current position within the
stream and the total number of passes over the stream. (index and
epoch)

For infinite streams the object will just maintain the index /
position within the stream.

In both cases it is necessary for the object to track the "step" size
(i.e. mini-batch size).  Additionally, because the data stream will be
accessed in parallel, it is necessary to track the position of each
rank within the stream in terms of offset.

..
   Data source class file:  The data source class tracks the statefule
   aspects of one logical stream of data.
   Data sources are either bounded or infinite
   data sources.  The class is responsible for keeping track of state
   with respect to


Sample list:

Track how to retrive a data set from the outside world.  This
typically is a set of file locations for each sample as well as a
count of how many samples are in the set.

Data coordinator:

Responsible for managing one or more data streams for each execution
context.  It is


data reader / loader:

Function to ingest bits from outside and place them into an in-memory
object that is managed by the data coordinator.

Data store:
in-memory data repository for holding samples that have been read in

io_data_buffer:
Holds sample being fetched or the future of it.

data packer:
copies data fields from conduit nodes and maps them to Hydrogen
matrices.  Specific to a data set

Data Set:

Composed of:
 - data reader
 - data stream
 - sample list
 - data packer
