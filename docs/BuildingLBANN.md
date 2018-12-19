## Building LBANN
### Dependencies

The following third-party packages are currently required to build
LBANN. All may be installed using
[spack](https://github.com/llnl/spack):

+ A C++11-compliant compiler.
+ OpenMP, version 3.0 or newer.
+ An MPI-3.0 implementation.
+ [CMake](https://cmake.org), version 3.9 or newer.
+ [CNPY](https://github.com/rogersce/cnpy.git)
+ [OpenCV](https://github.com/opencv/opencv)
++ [JPEG-turbo](https://github.com/libjpeg-turbo/libjpeg-turbo)
+ [ProtoBuf](https://github.com/protocolbuffers/protobuf)

The LLNL-maintained fork of Elemental,
[Hydrogen](https://github.com/llnl/elemental), is also required and
may also be installed via [spack](https://github.com/llnl/spack) using
the package name "hydrogen".

The LLNL-maintained communcation library,
[Aluminum](https://github.com/llnl/aluminum), is strongly recommended
for both CPU-only and CPU-GPU builds. Aluminum requires
[HWLOC](https://www.open-mpi.org/projects/hwloc/) and HWLOC is also
recommended for LBANN. Both Aluminum and HWLOC can be installed using
[spack](https://github.com/llnl/spack).

The LLNL-maintained [CONDUIT](https://github.com/llnl/conduit) package
is also optionally supported.

Building LBANN to use NVIDIA GPUs also requires the [CUDA
Toolkit](https://developer.nvidia.com/cuda-toolkit) and
[cuDNN](https://developer.nvidia.com/cudnn), both freely distributed
by NVIDIA. Moreover, Aluminum and Hydrogen must have been built with
CUDA support enabled.

Finally, LBANN offers slightly enhanced profiling annotations for
VTune and NVPROF.

### Building with [Spack](https://github.com/llnl/spack)

bvanessen should write this section.

### Buidling with [CMake](https://cmake.org)

LBANN uses [CMake](https://cmake.org) for its build system. The
primary use-cases are on UNIX-based platforms. As such, the build is
tested regularly on Linux-based machines, occasionally on OSX, and
never on Windows machines.

#### LBANN CMake options
The following options are exposed in the CMake build system.

+ `LBANN_WITH_ALUMINUM` (Default: `OFF`): Use the Aluminum communication
  package. This will be set to `ON` automatically if Hydrogen was
  built with Aluminum.
  
+ `LBANN_WITH_CNPY` (Default: `ON`): Build with support for CNPY for reading
  Numpy data.

+ `LBANN_WITH_CUDA` (Default: `OFF`): Enable a CUDA-aware build. This will fail
  if Hydrogen has not been compiled with CUDA support.
  
+ `LBANN_WITH_CUDNN` (Default: `ON`): Must be `ON` for CUDA-aware builds. This option
  is deprecated and will be removed.

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

+ `LBANN_WARNINGS_AS_ERRORS` (Default: `OFF`): Developer use only. Developers
  are encouraged to build with this `ON`.
  
+ `LBANN_VERBOSE` (Default: `OFF`): This option is deprecated.

The following variables may also be set:

+ `LBANN_DATATYPE` (Default: `float`): The datatype to use for
  training. Currently this must be `float` or `double`.

#### Controlling dependencies
The following cache variables may be set with CMake to identify
dependencies that are not installed into "typical" locations that
CMake searches by default.

+ `OpenCV_DIR` or `OPENCV_DIR`: The path to _either_ the OpenCV
  installation prefix _or_ the OpenCVConfig.cmake file.
  
+ `Hydrogen_DIR` or `HYDROGEN_DIR`: The path to _either_ the Hydrogen
  installation prefix _or_ the HydrogenConfig.cmake file.
  
+ `Aluminum_DIR` or `ALUMINUM_DIR` or `AL_DIR`: The path to _either_
  the Aluminum installation prefix _or_ the AluminumConfig.cmake file.
  
+ `CNPY_DIR`: The path to the CNPY installation prefix.

+ `LBANN_CONDUIT_DIR` or `CONDUIT_DIR`: The path to _either_ the
  CONDUIT installation prefix _or_ the conduit.cmake file.
  
+ `HWLOC_DIR`: The path to the HWLOC installation prefix.
