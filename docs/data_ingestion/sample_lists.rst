.. _sec:sample-lists:

Sample Lists
============

A sample list is a text file that is used to describe how
many data samples are used in a LBANN run, and where they exist on
parallel file system.  The use of a sample list allows the user to
quickly change the data set used via a human readable and editable
format.  Figures :numref:`hdf5_inclusive` and :numref:`hdf5_exclusive`
contains sample list examples. Sample lists are formatted as follows.
The first line is contains a control field dictating the format of the
file.  The features that are being specified are if:

* Are there multiple samples per file
* If there are multiple samples per file, Is the list of samples per
  line describing the samples that are included (INCLUSION) or
  excluded (EXCLUSION) from the data set
* Is the 4th line of the sample list a path to a labels file
* Does the 2rd line of the sample list include the number of unused
  samples

Valid options are:

* ``SINGLE-SAMPLE`` - has unused sample field and path to the label file
* ``MULTI-SAMPLE_EXCLUSION`` - has unused sample field and path to the label file
* ``MULTI-SAMPLE_INCLUSION`` - has unused sample field and path to the label file
* ``MULTI-SAMPLE_INCLUSION_V2`` - Does not have the unused sample field
  and has no label header
* ``CONDUIT_HDF5_INCLUSION`` - has unused sample field and no path to the label file
* ``CONDUIT_HDF5_EXCLUSION`` - has unused sample field and no path to the label file

Example of CONDUIT_HDF5_* sample lists
--------------------------------------

For the the **CONDUIT_HDF5_\*** sample lists, the second line
contains: total number of samples to use (included samples); total
number of samples NOT used (excluded samples); number of hdf5
files. The third line contains the base directory, that is, the
directory in which your hdf5 files are located. This directory may
contain subdirectories.

The remaining lines contain: an hdf5 pathname (hence a file's complete
pathname is the third line concatentated with this pathname); number of
samples to use (included samples); number of samples to exclude.
Remaining entries on the line a listing of the sample IDs to either
include or exclude, depending on whether the first line contains
CONDUIT_HDF5_INCLUSION or CONDUIT_HDF5_EXCLUSION.

If you are using all or a majority of the samples, it's best to use the
EXCLUSION version. The generated sample lists assume you are using all
samples in all files. Hence, the inclusion version contains all sample
IDs from all hdf5 file. If you are using all samples from your hdf5 file
you can use either generated list as is; else, you will need to edit to
indicate which samples to include or exclude.


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
