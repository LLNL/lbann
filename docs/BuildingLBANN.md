# Building LBANN
## Download

LBANN source code can be obtained from the [Github
repo](https://github.com/LLNL/lbann).

## Dependencies

The following packages and tools are required to build LBANN. All
packages listed below may be installed using
[Spack](https://github.com/llnl/spack). See
<a href="#building-with-spack">below</a>
for more details on using Spack to build a complete LBANN
environment.

The following basic tools are **required**.

+ A C++11-compliant compiler.
+ OpenMP, version 3.0 or newer.
+ An MPI-3.0 implementation.
+ [CEREAL](https://github.com/USCiLab/cereal) is used to handle
  complex serialization tasks.
+ [CMake](https://cmake.org), version 3.9 or newer.

The following LLNL-maintained packages are **required**.

+ [Hydrogen](https://github.com/llnl/elemental) is a fork of the
  [Elemental](https://github.com/elemental/elemental) distributed
  dense linear-algebra library and it may be installed via
  [Spack](https://github.com/llnl/spack) using the package name
  "hydrogen". If CUDA support is enabled in Hydrogen, LBANN will
  inherit this support.

The following third-party packages are **required**.

+ [CNPY](https://github.com/rogersce/cnpy.git) is used to ingest data
  in NumPy format. In principle this should be optional, but at time
  of writing, LBANN will not build without it.
+ [OpenCV](https://github.com/opencv/opencv) is used to preprocess
  image data. For performance reasons, it is recommend to build OpenCV
  with [JPEG-turbo](https://github.com/libjpeg-turbo/libjpeg-turbo)
  for JPEG format support.
+ [ProtoBuf](https://github.com/protocolbuffers/protobuf) is used to
  express models in a portable format.

The following LLNL-maintained packages are **optional**.

+ [Aluminum](https://github.com/llnl/aluminum) is a
  communication library optimized for machine learning and interaction
  with GPUs. We cannot recommend its use strongly enough. It can be
  built using [Spack](https://github.com/llnl/spack).
+ [CONDUIT](https://github.com/llnl/conduit) is used to ingest
  structured data produced by scientific simulations.

The following third-party packages are **optional**.

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


## Building with [Spack](https://github.com/llnl/spack)

### Setup Spack and local base tools

1.  Download and install [Spack](https://github.com/llnl/spack).
    Additionally setup shell support as discussed
    [here](https://spack.readthedocs.io/en/latest/module_file_support.html#id2).

    ```bash
    . ${SPACK_ROOT}/share/spack/setup-env.sh
    ```

2.  Setup your compiler and external software environment. For example,
    on LLNL\'s LC machines, one might load the following modules:
    ```bash
    ml gcc/7.3.0 mvapich2/2.3 cuda/10.0.130 # Pascal
    ```
    or
    ```bash
    ml gcc/7.3.1 cuda/9.2.148 spectrum-mpi/rolling-release  # Lassen / Sierra
    ```

    + Note to unload unwanted modules you can execute `ml` with
      package names prepended with a dash, e.g.: `ml -intel`. To
      unload all currently loaded modules, use `ml purge`.

### Building & Installing LBANN as a user

This section is work in progress. For now, follow the developer
instructions below. We are working to simplify this process.

### Building & Installing LBANN as a developer

Developers of LBANN will often need to interact with the source code
and/or advanced configuration options for Aluminum, Hydrogen, and
LBANN while the other dependencies remain constant. The Spack
installation instructions below set up a Spack environment with the
remaining dependencies, requiring the developer to build Aluminum,
Hydrogen, and LBANN separately, by whatever means they choose.

1.  Establish a Spack environment and install software dependencies.
    Note that there are four environments to pick from along two axes:

    1. developers or users
    2. x86_64 and ppc64le

    For example if you are a developer and want to build the inside of
    the git repo use the following instructions:
    ```bash
    export LBANN_HOME=/path/to/lbann/git/repo
    export LBANN_BUILD_DIR=/path/to/a/build/directory
    export LBANN_INSTALL_DIR=/path/to/an/install/directory
    cd ${LBANN_BUILD_DIR}
    spack env create -d . ${LBANN_HOME}/spack_environments/developer_release_<arch>_cuda_spack.yaml # where <arch> = x86_64 | ppc64le
    spack install
    spack env loads # Spack creates a file named loads that has all of the correct modules
    source loads
    unset LIBRARY_PATH
    ```

    + Note that the environments provided here have a set of external
      packages and compilers that are installed on an LLNL LC CZ
      system.  Please update these for your system environment.
      Alternatively, you can create baseline versions of the
      user-level Spack configuration files and remove the externals
      and compilers from the `spack.yaml` file. More details are
      provided [here](spack_environment.md).

    + Note that the initial build of all of the standard packages in Spack
      will take a while.

    + Note that the Spack module files set the `LIBRARY_PATH` environment
      variable. This behavior allows autotools-based builds to pickup the
      correct libraries but interferes with the way that CMake sets up
      RPATHs.  To correctly establish the RPATH, please unset the variable
      as noted above, or you can explicitly pass the RPATH fields to CMake
      using a command such as:
      ```bash
      cmake -DCMAKE_INSTALL_RPATH=$(sed 's/:/;/g' <<< "${LIBRARY_PATH}") \
            -DCMAKE_BUILD_RPATH=$(sed 's/:/;/g' <<< "${LIBRARY_PATH}") \
            ...
      ```

2.  Build LBANN locally from source and build Hydrogen and Aluminum
    using the superbuild. See
    <a href=#building-an-entire-ecosystem-with-the-superbuild>below</a>
    for a list and descriptions of all CMake flags known to LBANN's
    "Superbuild" build system. A representative CMake command line
    that expects `LBANN_HOME`, `LBANN_BUILD_DIR`, `LBANN_INSTALL_DIR`
    environment variables might be:
    ```bash
    cd ${LBANN_BUILD_DIR}
    cmake \
      -G Ninja \
      -D CMAKE_BUILD_TYPE:STRING=Release \
      -D CMAKE_INSTALL_PREFIX:PATH=${LBANN_INSTALL_DIR} \
      \
      -D LBANN_SB_BUILD_ALUMINUM=ON \
      -D ALUMINUM_ENABLE_MPI_CUDA=OFF \
      -D ALUMINUM_ENABLE_NCCL=ON \
      \
      -D LBANN_SB_BUILD_HYDROGEN=ON \
      -D Hydrogen_ENABLE_ALUMINUM=ON \
      -D Hydrogen_ENABLE_CUB=ON \
      -D Hydrogen_ENABLE_CUDA=ON \
      \
      -D LBANN_SB_BUILD_LBANN=ON \
      -D LBANN_DATATYPE:STRING=float \
      -D LBANN_SEQUENTIAL_INITIALIZATION:BOOL=OFF \
      -D LBANN_WITH_ALUMINUM:BOOL=ON \
      -D LBANN_WITH_CONDUIT:BOOL=ON \
      -D LBANN_WITH_CUDA:BOOL=ON \
      -D LBANN_WITH_CUDNN:BOOL=ON \
      -D LBANN_WITH_NCCL:BOOL=ON \
      -D LBANN_WITH_NVPROF:BOOL=ON \
      -D LBANN_WITH_SOFTMAX_CUDA:BOOL=ON \
      -D LBANN_WITH_TOPO_AWARE:BOOL=ON \
      -D LBANN_WITH_TBINF=OFF \
      -D LBANN_WITH_VTUNE:BOOL=OFF \
      ${LBANN_HOME}/superbuild

    ninja
    ```

The complete documentation for building LBANN directly with CMake can
be found [here](BuildingLBANNWithCMake.md).
