## Building LBANN
### Dependencies

The following third-party packages are currently required to build
LBANN. All may be installed using
[spack](https://github.com/llnl/spack):

+ A C++11-compliant compiler.
+ OpenMP, version 3.0 or newer.
+ An MPI-3.0 implementation.
+ [CMake](https://cmake.org), version 3.9 or newer.
+ [CNPY](https://github.com/rogersce/cnpy.git) is used to ingest data
  in NumPy format. In principle this should be optional, but at time
  of writing, LBANN will not build without it.
+ [OpenCV](https://github.com/opencv/opencv) is used to preprocess
  image data. For performance reasons, it is recommend to build OpenCV
  with [JPEG-turbo](https://github.com/libjpeg-turbo/libjpeg-turbo)
  for JPEG format support.
+ [ProtoBuf](https://github.com/protocolbuffers/protobuf) is used to
  express models in a portable format.

The following third-party packages are optional.

+ [CUDA](https://developer.nvidia.com/cuda-toolkit). The development
  team currently uses CUDA version 9.2. Building with CUDA support
  requires that Hydrogen has been built with CUDA support (see below).
  + [cuDNN](https://developer.nvidia.com/cudnn) is required if
    building LBANN with CUDA support. It is freely available as a binary
    distribution from NVIDIA.
+ [HWLOC](https://www.open-mpi.org/projects/hwloc/). HWLOC enables
  LBANN to make certain optimizations based on the hardware
  topology. Its use is strongly recommended.
+ NVTX. LBANN supports some improved annotations for NVPROF using
  NVTX. NVTX is provided as part of the CUDA toolkit.
+ VTune. LBANN supports some improved annotations for VTune.

The following LLNL-maintained packages are required.

+ [Hydrogen](https://github.com/llnl/elemental) is a fork of the
  Elemental distributed dense linear-algebra library and it may be
  installed via [spack](https://github.com/llnl/spack) using the
  package name "hydrogen". If CUDA support is required in LBANN,
  Hydrogen must also be build with CUDA support.

The following LLNL-maintained packages are optional.

+ [Aluminum](https://github.com/llnl/aluminum) is a
  communication library optimized for machine learning and interaction
  with GPUs. We cannot recommend its use strongly enough. It can be
  built using [spack](https://github.com/llnl/spack).
+ [CONDUIT](https://github.com/llnl/conduit) is used to ingest
  structured data produced by scientific simulations.


### Building with [Spack](https://github.com/llnl/spack)

Some variation on the theme of `spack install lbann`. bvanessen should
document the spack flags or whatever they call them that LBANN
supports.

### Buidling with [CMake](https://cmake.org)

LBANN uses [CMake](https://cmake.org) for its build system and a
version newer than or equal to 3.9.0 is required. LBANN development is
done primarily on UNIX-based platforms. As such, the build is tested
regularly on Linux-based machines, occasionally on OSX, and never on
Windows machines.

It is required that LBANN be built out-of-source. That is, CMake must
not be invoked in a directory containing a CMakeLists.

#### LBANN CMake options
The following options are exposed in the CMake build system.

+ `LBANN_WITH_ALUMINUM` (Default: `OFF`): Use the Aluminum communication
  package. This will be set to `ON` automatically if Hydrogen was
  built with Aluminum.

+ `LBANN_WITH_CNPY` (Default: `ON`): Build with support for CNPY for reading
  Numpy data.

+ `LBANN_WITH_CUDA` (Default: `OFF`): Enable a CUDA-aware build. If
  this flag is `ON` and Hydrogen CUDA support cannot be detected,
  LBANN's configuration step will fail.

+ `LBANN_WITH_CONDUIT` (Default: `OFF`): Build with support for CONDUIT.

+ `LBANN_WITH_NVPROF` (Default: `OFF`): Build with extra annotations for NVPROF.

+ `LBANN_WITH_TOPO_AWARE` (Default: `ON`): Use HWLOC for topology-aware choices.

+ `LBANN_WITH_TBINF` (Default: `ON`): Enable the Tensorboard interace.

+ `LBANN_WITH_VTUNE` (Default: `OFF`): Build with extra annotations for VTune.

+ `LBANN_DETERMINISTIC` (Default: `OFF`): Force as much of the code as possible
  to be deterministic. This is not a guarantee as certain operations
  in third-party libraries cannot be forced into a deterministic mode,
  especially for CUDA-enabled builds.

+ `LBANN_SEQUENTIAL_INITIALIZATION` (Default: `OFF`): Force sequentially
  consistent initialization of data structures.

+ `LBANN_WARNINGS_AS_ERRORS` (Default: `OFF`): Promote compiler
  warnings to errors. This should be used by developers
  only. Developers are encouraged to build with this `ON` prior to
  merging any code into the repository.

+ `LBANN_USE_PROTOBUF_MODULE` (Default: `OFF`): Search for Protobuf
  using CMake's `FindProtobuf.cmake` module instead of the Protobuf
  config file. This is useful on platforms with differently
  architected compute nodes or when the config method is inexplicably
  failing.

The following variables may also be set:

+ `LBANN_DATATYPE` (Default: `float`): The datatype to use for
  training. Currently this must be `float` or `double`.

#### Controlling dependency resolution
The following variables may be set with CMake to identify dependencies
that are not installed into the "typical" locations that CMake
searches by default. They may be either exported into the environment
used by CMake using whatever mechanisms are allowed by the shell or
passed to CMake as a cache variable
(e.g., `cmake -DPKG_DIR=/path/to/pkg`).
The latter option is recommended.

+ `Aluminum_DIR` or `ALUMINUM_DIR` or `AL_DIR`: The path to _either_
  the Aluminum installation prefix _or_ the AluminumConfig.cmake
  file. If Hydrogen has not been built with Aluminum support, set
  `LBANN_WITH_ALUMINUM=ON` to enable Aluminum support.
+ `CNPY_DIR`: The path to the CNPY installation prefix. Must set
  `LBANN_WITH_CNPY=ON` to enable CNPY support.
+ `CONDUIT_DIR` or `CONDUIT_DIR`: The path to _either_ the
  CONDUIT installation prefix _or_ the conduit.cmake file. Must set
  `LBANN_WITH_CONDUIT=ON` to enable CONDUIT support.
  + `HDF5_DIR`: The path to _either_ the HDF5 installation prefix _or_
    the hdf5_config.cmake file. There is a known issue with CONDUIT
    that it may link to HDF5 but not properly export that dependency.
+ `HWLOC_DIR`: The path to the HWLOC installation prefix. Must set
  `LBANN_WITH_HWLOC=ON` to enable HWLOC support.
+ `Hydrogen_DIR` or `HYDROGEN_DIR`: The path to _either_ the Hydrogen
  installation prefix _or_ the HydrogenConfig.cmake file.
+ `NVTX_DIR`: The path the the prefix of NVTX. This should not be used
  except in circumstances in which one might want to link to a
  different NVTX installation than the CUDA toolkit. Under normal
  circumstances, if CUDA was found without issue, NVTX should be as
  well.
+ `OpenCV_DIR` or `OPENCV_DIR`: The path to _either_ the OpenCV
  installation prefix _or_ the OpenCVConfig.cmake file.
+ `Protobuf_DIR` or `PROTOBUF_DIR`: The path to _either_ the Protobuf
  installation prefix _or_ the protobuf-config.cmake file.
+ `VTUNE_DIR`: The path to the prefix of the VTune (or Intel compiler
  suite) installation.

Compilers, include CUDA compilers, are found using the default CMake
mechanisms, as are OpenMP and MPI. Thus, the process of finding these
tools can be manipulated using the usual CMake mechanisms and/or cache
variables as [documented by CMake](https://cmake.org/documentation).

Except where otherwise noted, this list attempts to address the first
level of dependencies of LBANN, that is, those that are one edge away
in the DAG. If deeper dependency issues appear, please consult the
documentation of the packages that are causing the issues as they may
require additional CMake/environment flags to be set before properly
resolving.

#### Example CMake invocation
A sample CMake build for LBANN might look like the following.

    cmake \
      -D LBANN_WITH_CUDA:BOOL=ON \
      -D LBANN_WITH_NVPROF:BOOL=ON \
      -D LBANN_DATATYPE:STRING=float \
      -D Hydrogen_DIR:PATH=/path/to/hydrogen \
      -D HWLOC_DIR:PATH=/path/to/hwloc \
      /path/to/lbann

### Building an entire ecosystem with the "Superbuild"

__WARNING__: This is primarily for developer convenience and is not
meant to be robust to all possible use-cases for LBANN.

LBANN includes CMake `ExternalProject` definitions for a large portion
of its dependency graph. The following dependencies are
supported. These are one or two edges from LBANN in the
dependency DAG.

+ Aluminum
+ CNPY
+ CONDUIT
+ [CUB](https://github.com/nvlabs/cub). This is used by Hydrogen for
  efficiently managing GPU memory.
+ [HDF5](https://www.hdfgroup.org/solutions/hdf5). This is a
  dependency of CONDUIT.
+ Hydrogen
+ [JPEG-turbo](https://github.com/libjpeg-turbo/libjpeg-turbo). This
  is a dependency of OpenCV.
+ [OpenBLAS](https://github.com/xianyi/OpenBLAS.git). This is an
  optional dependency of Hydrogen. It is recommended if your system
  does not have a system-optimized BLAS distribution (e.g., Intel's MKL).
+ OpenCV
+ Protobuf

The following dependencies are known to exist but for some reason or
another are not supported by the superbuild framework.

+ cuDNN is a freely available binary package available from NVIDIA.
+ NCCL is a freely available binary package available from
  NVIDIA. Inspired users may also build it from source from its
  [github repository](https://github.com/nvidia/nccl).
+ HWLOC is often installed by default, especially on large
  supercomputers. Certain components may require superuser access to
  configure, but these features are not used by LBANN. If it is not
  available, ask the system administrators, consult the package
  manager, install using Spack, or build from
  [source](https://www.open-mpi.org/projects/hwloc/).

The superbuild system is itself a CMake project rooted in
`$LBANN_HOME/superbuild` (distinct from the LBANN CMake project rooted
in `$LBANN_HOME`). Options that control the superbuild system are
prefixed with `LBANN_SB_`; other options that appear in a CMake
invocation for the superbuild are either interpreted on a sub-project
basis or forwarded to certain sub-projects.

#### Choosing packages to build in the Superbuild
The superbuild system is _constructive_ or _additive_; that is, it
will only build the packages that it is asked to build. Any required
package that is not requested is assumed to exist on the system by the
time it is needed by whichever package requires it. For example, if
HDF5 is provided by the system administrators on a system, it does not
need to be built and CONDUIT can be built by pointing its build to the
system HDF5.

Packages are included in a superbuild by passing
`LBANN_SB_BUILD_<PKG>` options to CMake _for each package_ that it
should build, including LBANN itself. E.g.,

    cmake \
      -DLBANN_SB_BUILD_ALUMINUM=ON \
      -DLBANN_SB_BUILD_HYDROGEN=ON \
      -DLBANN_SB_BUILD_LBANN=ON \
      /path/to/lbann/superbuild

will invoke the superbuild to build Aluminum, Hydrogen, and LBANN
_only_. Acceptable values for `<PKG>` are `ALUMINUM`, `CNPY`,
`CONDUIT`, `CUB`, `HDF5`, `HYDROGEN`, `JPEG_TURBO`, `OPENCV`,
`PROTOBUF` and `LBANN`.

#### Forwarding options to sub-projects
`LBANN_SB_FWD_<PKG>_<OPTION>=...`
