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

lbann_sb_init_extern_pkg(
  NAME hwloc
  LANGUAGES C CXX
  GITHUB_URL open-mpi/hwloc
  GIT_TAG "master")

# Allow interested users to build doxygen documentation
lbann_sb_this_pkg_option(
  BUILD_DOXYGEN
  "Build doxygen documentation?"
  OFF)

if (LBANN_SB_FWD_hwloc_BUILD_DOXYGEN)
  set(_hwloc_doxygen_opt "--enable-doxygen")
else ()
  set(_hwloc_doxygen_opt "--disable-doxygen")
endif ()

# Prefer a user-specified ROCM path, then check ROCM_PATH.
if (LBANN_SB_FWD_hwloc_ROCM_PATH)
  set(_hwloc_rocm_path_opt
    "--with-rocm=${LBANN_SB_FWD_HWLOC_ROCM_PATH}")
elseif (DEFINED ENV{ROCM_PATH})
  set(_hwloc_rocm_path_opt
    "--with-rocm=$ENV{ROCM_PATH}")
endif ()

# Prefer a user-specified CUDA path, then check CUDA_HOME
if (LBANN_SB_FWD_hwloc_CUDA_PATH)
  set(_hwloc_cuda_path_opt
    "--with-cuda=${LBANN_SB_FWD_HWLOC_CUDA_PATH}")
elseif (DEFINED ENV{CUDA_HOME})
  set(_hwloc_cuda_path_opt
    "--with-cuda=$ENV{CUDA_HOME}")
endif ()

# This is an autotools package, so we need 'make'.
find_program(GNU_MAKE_PROGRAM make)

include (ExternalProject)
ExternalProject_Add(${PKG_NAME}
  PREFIX "${CMAKE_CURRENT_BINARY_DIR}"
  ${LBANN_SB_GIT_REPOSITORY_TAG} ${LBANN_SB_${PKG_NAME}_URL}
  ${LBANN_SB_GIT_TAG_TAG} ${LBANN_SB_${PKG_NAME}_TAG}
  TMP_DIR "${CMAKE_CURRENT_BINARY_DIR}/tmp"
  STAMP_DIR "${CMAKE_CURRENT_BINARY_DIR}/stamp"

  SOURCE_DIR "${LBANN_SB_${PKG_NAME}_SOURCE_DIR}"
  INSTALL_DIR "${LBANN_SB_${PKG_NAME}_PREFIX}"

  BUILD_IN_SOURCE 1
  USES_TERMINAL_BUILD 1
  LOG_DOWNLOAD 1
  LOG_UPDATE 1
  LOG_CONFIGURE 1
  LOG_BUILD 1
  LOG_INSTALL 1
  LOG_TEST 1

  CONFIGURE_COMMAND
  "${LBANN_SB_${PKG_NAME}_SOURCE_DIR}/configure"
  "--prefix=${LBANN_SB_${PKG_NAME}_PREFIX}"
  "CC=${LBANN_SB_${PKG_NAME}_C_COMPILER}"
  "CXX=${LBANN_SB_${PKG_NAME}_CXX_COMPILER}"
  ${_hwloc_doxygen_opt}
  ${_hwloc_rocm_path_opt}
  ${_hwloc_cuda_path_opt}

  BUILD_COMMAND
  ${GNU_MAKE_PROGRAM} -j${${PKG_NAME}_MAX_MAKE_JOBS}
  INSTALL_COMMAND
  ${GNU_MAKE_PROGRAM} install
)

ExternalProject_Add_Step(${PKG_NAME} autogen
  COMMAND "<SOURCE_DIR>/autogen.sh"
  COMMENT "Running autogen.sh for ${PKG_NAME}"
  DEPENDEES download
  DEPENDERS configure
  INDEPENDENT YES
  BYPRODUCTS "<SOURCE_DIR>/configure"
  ALWAYS NO
  WORKING_DIRECTORY "<SOURCE_DIR>"
  LOG YES
  USES_TERMINAL NO)

set(${PKG_NAME}_DIR ${LBANN_SB_${PKG_NAME}_PREFIX}
  CACHE INTERNAL "The install prefix of ${PKG_NAME}.")
