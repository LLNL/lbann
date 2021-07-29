.. _sec:sample-lists:

Sample Lists
============

A sample list is a text file that is used to describe how
many data samples are used in a LBANN run and where they exist on
parallel file system.  The use of a sample list allows the user to
quickly change the data set used via a human readable and editable
format. :numref:`hdf5_inclusive` and :numref:`hdf5_exclusive`
contain sample list examples.

Sample lists are formatted as described below.

The first line contains a control field dictating the format of the
file.  The features that are being specified are:

* Are there multiple samples per file?
* If there are multiple samples per file, is the list of samples per
  line describing the samples that are included (INCLUSION) or
  excluded (EXCLUSION) from the data set?
* Is the fourth line of the sample list a path to a labels file?
* Does the second line of the sample list include the number of unused
  samples?

Valid options are as follows:

* ``SINGLE-SAMPLE`` - has unused sample field and path to the label
  file.
* ``MULTI-SAMPLE_EXCLUSION`` - has unused sample field and path to the
  label file.
* ``MULTI-SAMPLE_INCLUSION`` - has unused sample field and path to the
  label file.
* ``MULTI-SAMPLE_INCLUSION_V2`` - Does not have the unused sample
  field and no label header.
* ``CONDUIT_HDF5_INCLUSION`` - has unused sample field and no path to
  the label file.
* ``CONDUIT_HDF5_EXCLUSION`` - has unused sample field and no path to
  the label file.

The format of the remaining lines is dependent on the sample list type
specified in the first line. Below we describe the most common format
that users will interact with, the ``CONDUIT_HDF5_INCLUSION`` and
``CONDUIT_HDF5_EXCLUSION`` formats. We are working on documentation
for the other formats.


Example of CONDUIT_HDF5_* sample lists
--------------------------------------

For the the ``CONDUIT_HDF5_*`` sample lists, the second line
contains the following fields, delimited by a space:

* The total number of samples to use (included samples).
* The total number of samples NOT used (excluded samples).
* The total number of HDF5 files in the data set.

The third line contains the base directory. This is the top-level
directory under which the HDF5 files are located. This directory may
contain subdirectories.

The remaining lines contain the following fields, delimited by a
space:

* An HDF5 pathname, relative to the base directory specified on the
  third line.
* The number of samples to use (included samples).
* The number number of samples to exclude.

Any additional entries on the line are the sample IDs to either
include or exclude, depending on whether the first line contains
``CONDUIT_HDF5_INCLUSION`` or ``CONDUIT_HDF5_EXCLUSION``,
respectively.

To use all or a majority of the samples, it is best to use the
``EXCLUSION`` version. The generated sample lists assume the use of
all samples in all files. Hence, the inclusion version contains all
sample IDs from all HDF5 files. To use all samples from an HDF5 file,
the generated sample list may be used directly. Otherwise, the sample
list will need to be manually edited to indicate which samples to
include or to exclude.


.. code-block:: bash
   :caption: Example of a inclusive sample list with 7 valid samples
             from 3 files.  Note that 23 samples in the 3 files were
             excluded.
   :name: hdf5_inclusive

   CONDUIT_HDF5_INCLUSION
   7 23 3
   /p/vast1/data
   file_1.h5 3 7 runid/002 runid/005 runid/011
   file_2.h5 2 8 runid/005 runid/006
   file_3.h5 2 8 runid/000 runid/002


.. code-block:: bash
   :caption: Example of an exclusive sample list with 61 valid samples
             from 3 files and 3 excluded samples.
   :name: hdf5_exclusive

   CONDUIT_HDF5_EXCLUSION
   61 3 3
   /p/vast1/lbann/datasets/PROBIES/h5_data/
   h5out_1.h5 18 2 RUN_ID/000000003 RUN_ID/000000021
   h5out_2.h5 24 0
   h5out_3.h5 19 1 RUN_ID/000000003
