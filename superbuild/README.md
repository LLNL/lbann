# WARNING: NOT A PACKAGE MANAGER

This SuperBuild is provided as a convenience for users and developers
who wish to get started quickly with LBANN. This is **not** a replacement
for a legitimate package manager like
[spack](https://github.com/llnl/spack). However, we will do our best
to maintain this as a suitable way to build LBANN on workstations, as
well as the LLNL clusters.

# LBANN SuperBuild

Welcome to the LBANN SuperBuild. The purpose of this CMake framework
is to build third-party library dependencies of LBANN as well as LBANN
itself. The build system is exclusive by default; i.e., packages,
including LBANN, will not be built unless required. If a third-party
library is not built by this project, it must be provided via a
forwarded argument (see [Passing options to SuperBuild
Subpackages](#passing-options-to-superbuild-subpackages)).

## SuperBuild CMake Options

Arguments that are processed by the SuperBuild are prefixed with
`LBANN_SB_`. The notable options are:

- `LBANN_SB_BUILD_<PKG>`, where `<PKG>` is one of `CNPY, CUB,
  ELEMENTAL, HYDROGEN, JPEG_TURBO, LBANN, OPENBLAS, OPENCV,
  PROTOBUF`. Enables the build of `<PKG>` as a subproject.

- `LBANN_SB_FWD_*`. See [Passing options to SuperBuild
  Subpackages](#passing-options-to-superbuild-subpackages)).

- `LBANN_SB_CLONE_CLEAN_LBANN`. If `ON`, the SuperBuild will clone a
  fresh git repository and use that for the build. If `OFF` (default),
  it will use the source directory located one level up from this
  directory.

- `<PKG>_URL`. The URL from which to clone `<PKG>`.

- `<PKG>_TAG`. The Git tag, hash, or branch to checkout for `<PKG>`.

## Passing options to SuperBuild Subpackages

Certain options passed to the SuperBuild will get forwarded to
subpackages. The syntax for achieving this is described here.

- `<PKG>_<OPTION>`. Certain packages format their CMake options this
  way by default (e.g., LBANN). These options are forwarded as-is,
  i.e., as `-D<PKG>_<OPTION>:<ARG_TYPE>=${<PKG>_<OPTION>}`, where
  `<ARG_TYPE>` is computed by CMake. Other packages do not format
  their CMake options this way (e.g., OpenCV). These options are
  forwarded with the `<PKG>_` prefix removed, i.e., as
  `-D<OPTION>:<ARG_TYPE>=${<PKG>_<OPTION>}`.

- `<PKG>_<CMAKE_OPTION>`. `<CMAKE_OPTION>` is any option that starts
  with the word `CMAKE`, e.g., `CMAKE_C_COMIPLER` or
  `CMAKE_BUILD_TYPE`. These options are stripped of the `<PKG>_`
  prefix and forwarded, i.e.,
  `-D<CMAKE_OPTION>:<ARG_TYPE>=${<PKG>_<CMAKE_OPTION>}`.

- `LBANN_SB_FWD_<PKG>_<OPTION>`. These options are forwarded to
  `<PKG>` as `-D<OPTION>:<ARG_TYPE>=${LBANN_SB_FWD_<PKG>_<OPTION>}`.

Some subpackages (e.g., Protobuf) are not CMake packages. There is not
(yet) a uniform way to forward options to these
packages. `<PKG>_CMAKE_INSTALL_PREFIX` gets forwarded to the prefix
mechanism of the underlying build system and
`<PKG>_CMAKE_{C,CXX,Fortran}_COMPILER` gets forwarded to the compiler
setting mechanism of the underlying build system. Other options are
documented on a per-package basis.

## A note about CUDA support

If your preferred host compiler for NVCC is not set in your
environment, CMake will probably get it wrong. If you build a package
with CUDA using NVCC, you should be sure to pass in
`CMAKE_CUDA_HOST_COMPILER` or set
`CMAKE_CUDA_FLAGS=-ccbin=/path/to/host/compiler`.

## Sample invocation of CMake

A sample invocation of CMake that builds a CPU-based LBANN build with
all default options is:

```
cd $LBANN_HOME
mkdir build
cd build
cmake $LBANN_HOME/superbuild -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/path/to/install -DLBANN_SB_BUILD_CNPY=ON -DLBANN_SB_BUILD_HYDROGEN=ON -DLBANN_SB_BUILD_OPENCV=ON -DLBANN_SB_BUILD_PROTOBUF=ON -DLBANN_SB_BUILD_LBANN=ON -DCMAKE_C_COMPILER=$(which clang) -DCMAKE_CXX_COMPILER=$(which clang++) -DCMAKE_Fortran_COMPILER=$(which gfortran)
```

