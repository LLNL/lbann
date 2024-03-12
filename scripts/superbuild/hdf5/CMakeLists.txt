################################################################################
## Copyright (c) 2014-2024, Lawrence Livermore National Security, LLC.
## Produced at the Lawrence Livermore National Laboratory.
## Written by the LBANN Research Team (B. Van Essen, et al.) listed in
## the CONTRIBUTORS file. <lbann-dev@llnl.gov>
##
## LLNL-CODE-697807.
## All rights reserved.
##
## This file is part of LBANN: Livermore Big Artificial Neural Network
## Toolkit. For details, see http://software.llnl.gov/LBANN or
## https://github.com/LLNL/LBANN.
##
## Licensed under the Apache License, Version 2.0 (the "Licensee"); you
## may not use this file except in compliance with the License.  You may
## obtain a copy of the License at:
##
## http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
## implied. See the License for the specific language governing
## permissions and limitations under the license.
################################################################################
# These defaults are set so as to be most amenable to LBANN(/Conduit).
set(LBANN_SB_FWD_HDF5_DEFAULT_API_VERSION "v112"
  CACHE STRING "Default API version for HDF5")

option(LBANN_SB_FWD_HDF5_BUILD_TESTING
  "Build HDF5 tests."
  OFF)

# HDF5-specific configuration options explicitly exposed
option(LBANN_SB_FWD_HDF5_HDF5_BUILD_EXAMPLES
  "Build HDF5 examples." OFF)
option(LBANN_SB_FWD_HDF5_HDF5_ENABLE_PARALLEL
  "Build HDF5 with parallel (MPI) support." ON)

# option(LBANN_SB_FWD_HDF5_HDF5_USE_16_API_DEFAULT
#   "Use 1.6 API by default"
#   OFF)
# option(LBANN_SB_FWD_HDF5_HDF5_USE_18_API_DEFAULT
#   "Use 1.8 API by default"
#   OFF)
# option(LBANN_SB_FWD_HDF5_HDF5_USE_110_API_DEFAULT
#   "Use 1.10 API by default"
#   OFF)
# option(LBANN_SB_FWD_HDF5_HDF5_USE_112_API_DEFAULT
#   "Use 1.12 API by default"
#   ON)
option(LBANN_SB_FWD_HDF5_HDF5_BUILD_FORTRAN
  "Build HDF5 with fortran support"
  OFF)
option(LBANN_SB_FWD_HDF5_HDF5_GENERATE_HEADERS
  "See HDF5 docs."
  ON)
option(LBANN_SB_FWD_HDF5_ONLY_SHARED_LIBS
  "Only Build Shared Libraries."
  ON)
# option(LBANN_SB_FWD_HDF5_DEFAULT_API_VERSION
#   "Only Build Shared Libraries."
#   v112)

# At present, this is required for LBANN/JAG use.
option(LBANN_SB_FWD_HDF5_HDF5_ENABLE_Z_LIB_SUPPORT
  "Build HDF5 with ZLIB support"
  ON)

set(MAYBE_HDF5_FORTRAN)
if (LBANN_SB_FWD_HDF5_HDF5_BUILD_FORTRAN)
  set(MAYBE_HDF5_FORTRAN Fortran)
endif ()

if (LBANN_SB_FWD_HDF5_HDF5_ENABLE_PARALLEL)
  # The C++ and Parallel paths are mutually exclusive, apparently.
  set(LBANN_SB_FWD_HDF5_HDF5_BUILD_CPP_LIB OFF CACHE BOOL "" FORCE)
endif ()

lbann_sb_add_cmake_extern_pkg(
  NAME HDF5
  LANGUAGES C CXX ${MAYBE_HDF5_FORTRAN}
  GITHUB_URL HDFGroup/hdf5
  GIT_TAG "hdf5-1_12_3")

set(HDF5_DIR ${LBANN_SB_HDF5_PREFIX}
  CACHE INTERNAL "The install prefix of HDF5.")
