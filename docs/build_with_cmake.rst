.. role:: cxx(code)
   :language: cpp

.. _build-with-cmake:

==================================================
Building LBANN with `CMake <https://cmake.org>`_
==================================================

LBANN uses `CMake <https://cmake.org>`_ for its build system and a
version newer than or equal to 3.12.0 is required. LBANN development is
done primarily on UNIX-based platforms. As such, the build is tested
regularly on Linux-based machines, occasionally on OSX, and never on
Windows machines.

The CMake build system is available to any users or developers who
need a more fine-grained level of control over dependency resolution
and/or features of LBANN. The LBANN team has made an effort to expose
as many knobs as possible through the Spack package but if something
is missing, please `open an issue <https://github.com/LLNL/lbann/issues/new>`_.

It is required that LBANN be built out-of-source. That is, CMake must
not be invoked in a directory containing a CMakeLists.

--------------------
Dependencies
--------------------

The following packages and tools are required to build LBANN. All
packages listed below may be installed using `Spack
<https://github.com/llnl/spack>`_. See :ref:`the Spack installation
instructions <building-with-spack>` for more details on using Spack to
build a complete LBANN environment.

The following basic tools are **required**.

+ A C++11-compliant compiler.

+ OpenMP, version 3.0 or newer.

+ An MPI-3.0 implementation.

+ `CEREAL <https://github.com/USCiLab/cereal>`_ is used to handle
  complex serialization tasks.

+ `CMake <https://cmake.org>`_, version 3.12 or newer.

The following LLNL-maintained packages are **required**.

+ `Hydrogen <https://github.com/llnl/elemental>`_ is a fork of the
  `Elemental <https://github.com/elemental/elemental>`_ distributed
  dense linear-algebra library and it may be installed via
  `Spack <https://github.com/llnl/spack>`_ using the package name
  "hydrogen". If CUDA support is enabled in Hydrogen, LBANN will
  inherit this support.

The following third-party packages are **required**.

+ `CNPY <https://github.com/rogersce/cnpy.git>`_ is used to ingest data
  in NumPy format. In principle this should be optional, but at time
  of writing, LBANN will not build without it.

+ `OpenCV <https://github.com/opencv/opencv>`_ is used to preprocess
  image data. For performance reasons, it is recommend to build OpenCV
  with `JPEG-turbo <https://github.com/libjpeg-turbo/libjpeg-turbo>`_
  for JPEG format support.

+ `ProtoBuf <https://github.com/protocolbuffers/protobuf>`_ is used to
  express models in a portable format.

The following LLNL-maintained packages are **optional**.

+ `Aluminum <https://github.com/llnl/aluminum>`_ is a
  communication library optimized for machine learning and interaction
  with GPUs. We cannot recommend its use strongly enough. It can be
  built using `Spack <https://github.com/llnl/spack>`_.

+ `CONDUIT <https://github.com/llnl/conduit>`_ is used to ingest
  structured data produced by scientific simulations.

The following third-party packages are **optional**.

+ `CUDA <https://developer.nvidia.com/cuda-toolkit>`_. The development
  team currently uses CUDA version 9.2. Building with CUDA support
  requires that Hydrogen has been built with CUDA support (see below).

+ `cuDNN <https://developer.nvidia.com/cudnn>`_ is required if
  building LBANN with CUDA support. It is freely available as a binary
  distribution from NVIDIA.

+ `HWLOC <https://www.open-mpi.org/projects/hwloc/>`_. HWLOC enables
  LBANN to make certain optimizations based on the hardware
  topology. Its use is strongly recommended.

+ NVTX. LBANN supports some improved annotations for NVPROF using
  NVTX. NVTX is provided as part of the CUDA toolkit.

+ VTune. LBANN supports some improved annotations for VTune.



--------------------
LBANN CMake options
--------------------

The following options are exposed in the CMake build system.

+ :code:`LBANN_WITH_ALUMINUM` (Default: :code:`OFF`): Use the Aluminum communication
  package. This will be set to :code:`ON` automatically if Hydrogen was
  built with Aluminum.

+ :code:`LBANN_WITH_CNPY` (Default: :code:`ON`): Build with support for CNPY for reading
  Numpy data.

+ :code:`LBANN_WITH_CONDUIT` (Default: :code:`OFF`): Build with support for CONDUIT.

+ :code:`LBANN_WITH_NVPROF` (Default: :code:`OFF`): Build with extra annotations for NVPROF.

+ :code:`LBANN_WITH_TOPO_AWARE` (Default: :code:`ON`): Use HWLOC for topology-aware choices.

+ :code:`LBANN_WITH_TBINF` (Default: :code:`ON`): Enable the Tensorboard interace.

+ :code:`LBANN_WITH_VTUNE` (Default: :code:`OFF`): Build with extra annotations for VTune.

+ :code:`LBANN_DETERMINISTIC` (Default: :code:`OFF`): Force as much of the code as possible
  to be deterministic. This is not a guarantee as certain operations
  in third-party libraries cannot be forced into a deterministic mode,
  especially for CUDA-enabled builds.

+ :code:`LBANN_SEQUENTIAL_INITIALIZATION` (Default: :code:`OFF`): Force sequentially
  consistent initialization of data structures.

+ :code:`LBANN_WARNINGS_AS_ERRORS` (Default: :code:`OFF`): Promote compiler
  warnings to errors. This should be used by developers
  only. Developers are encouraged to build with this :code:`ON` prior to
  merging any code into the repository.

+ :code:`LBANN_USE_PROTOBUF_MODULE` (Default: :code:`OFF`): Search for
  Protobuf using CMake's :code:`FindProtobuf.cmake` module instead of
  the Protobuf config file. This is useful on platforms with
  differently architected compute nodes or when the config method is
  inexplicably failing.

The following variables may also be set:

+ :code:`LBANN_DATATYPE` (Default: :cxx:`float`): The datatype to use for
  training. Currently this must be :cxx:`float` or :cxx:`double`.

The following variable has been deprecated and removed:

+ :code:`LBANN_WITH_CUDA`. The "CUDA-ness" of LBANN is now tied 1:1 with the
  "CUDA-ness" of Hydrogen. At present, it seems like unnecessary
  overhead to support the situation in which Hydrogen has CUDA support
  but LBANN doesn't want to use it until a compelling use-case reveals
  itself.

-----------------------------------
Controlling dependency resolution
-----------------------------------

The following variables may be set with CMake to identify dependencies
that are not installed into the "typical" locations that CMake
searches by default. They may be either exported into the environment
used by CMake using whatever mechanisms are allowed by the shell or
passed to CMake as a cache variable
(e.g., :code:`cmake -DPKG_DIR=/path/to/pkg`).
The latter option is recommended.

+ :code:`Aluminum_DIR` or :code:`ALUMINUM_DIR` or :code:`AL_DIR`: The
  path to *either* the Aluminum installation prefix *or* the
  :code:`AluminumConfig.cmake` file. If Hydrogen has not been built
  with Aluminum support, set :code:`LBANN_WITH_ALUMINUM=ON` to enable
  Aluminum support.

+ :code:`CEREAL_DIR`: The path to *either* the CEREAL installation
  prefix *or* the :code:`cereal-config.cmake` file.

+ :code:`CNPY_DIR`: The path to the CNPY installation prefix. Must set
  :code:`LBANN_WITH_CNPY=ON` to enable CNPY support.

+ :code:`CONDUIT_DIR` or :code:`CONDUIT_DIR`: The path to *either* the
  CONDUIT installation prefix *or* the :code:`ConduitConfig.cmake`
  file. Must set :code:`LBANN_WITH_CONDUIT=ON` to enable CONDUIT
  support.

+ :code:`HDF5_DIR`: The path to *either* the HDF5 installation prefix
  *or* the :code:`hdf5_config.cmake` file. There is a known issue with
  CONDUIT that it may link to HDF5 but not properly export that
  dependency.

+ :code:`HWLOC_DIR`: The path to the HWLOC installation prefix. Must
  set :code:`LBANN_WITH_HWLOC=ON` to enable HWLOC support.

+ :code:`Hydrogen_DIR` or :code:`HYDROGEN_DIR`: The path to *either*
  the Hydrogen installation prefix *or* the
  :code:`HydrogenConfig.cmake` file.

+ :code:`NVTX_DIR`: The path the the prefix of NVTX. This should not
  be used except in circumstances in which one might want to link to a
  different NVTX installation than the CUDA toolkit. Under normal
  circumstances, if CUDA was found without issue, NVTX should be as
  well.

+ :code:`OpenCV_DIR` or :code:`OPENCV_DIR`: The path to *either* the
  OpenCV installation prefix *or* the :code:`OpenCVConfig.cmake`
  file.

+ :code:`Protobuf_DIR` or :code:`PROTOBUF_DIR`: The path to *either*
  the Protobuf installation prefix *or* the
  :code:`protobuf-config.cmake` file.

+ :code:`VTUNE_DIR`: The path to the prefix of the VTune (or Intel
  compiler suite) installation.

Compilers, include CUDA compilers, are found using the default CMake
mechanisms, as are OpenMP and MPI. Thus, the process of finding these
tools can be manipulated using the usual CMake mechanisms and/or cache
variables as `documented by CMake <https://cmake.org/documentation>`_.

Except where otherwise noted, this list attempts to address the first
level of dependencies of LBANN, that is, those that are one edge away
in the DAG. If deeper dependency issues appear, please consult the
documentation of the packages that are causing the issues as they may
require additional CMake/environment flags to be set before properly
resolving.

------------------------------
Example CMake invocation
------------------------------

A sample CMake build for LBANN might look like the following.

.. code-block:: bash

    cmake \
      -D LBANN_WITH_CUDA:BOOL=ON \
      -D LBANN_WITH_NVPROF:BOOL=ON \
      -D LBANN_DATATYPE:STRING=float \
      -D Hydrogen_DIR:PATH=/path/to/hydrogen \
      -D HWLOC_DIR:PATH=/path/to/hwloc \
      /path/to/lbann
