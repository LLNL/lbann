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
forwarded argument (see (#passing-options-to-superbuild-subpackages)).

## SuperBuild CMake Options

Arguments that are processed by the SuperBuild are prefixed with
`LBANN_SB_`. The notable options are:

- `LBANN_SB_BUILD_<PKG>`, where `<PKG>` is one of `CNPY, CUB,
  ELEMENTAL, HYDROGEN, JPEG_TURBO, LBANN, OPENBLAS, OPENCV,
  PROTOBUF`. Enables the build of `<PKG>` as a subproject.

- `LBANN_SB_FWD_*`. See (#passing-options-to-superbuild-subpackages).

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

- `<PKG>_<CMAKE_OPTION>`. These options are stripped of the `<PKG>_`
  prefix and forwarded, i.e.,
  `-D<CMAKE_OPTION>:<ARG_TYPE>=${<PKG>_<CMAKE_OPTION>}`.

- `LBANN_SB_FWD_<PKG>_<OPTION>`. These options are forwarded to
  `<PKG>` as `-D<OPTION>:<ARG_TYPE>=${LBANN_SB_FWD_<PKG>_<OPTION>}`.
