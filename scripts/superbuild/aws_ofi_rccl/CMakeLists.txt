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
  NAME AWS_OFI_RCCL
  LANGUAGES C
  GITHUB_URL ROCm/aws-ofi-rccl
  GIT_TAG "cxi"  # This is the head of their develop.
  DEPENDS_ON RCCL)

# Prefer a user-specified HIP path, then check ROCM_PATH.
if (LBANN_SB_FWD_AWS_OFI_RCCL_HIP_PATH)
  set(_aws_ofi_rccl_hip_path_opt
    "--with-hip=${LBANN_SB_FWD_AWS_OFI_RCCL_HIP_PATH}")
elseif (DEFINED ENV{ROCM_PATH})
  set(_aws_ofi_rccl_hip_path_opt
    "--with-hip=$ENV{ROCM_PATH}")
else ()
  message(WARNING
    "You have enabled AWS_OFI_RCCL package, but ROCM_PATH "
    "is not available in your environment.")
endif ()

# First look for a user-provided path. Then look for a RCCL built by
# this superbuild. Finally, land back on the ROCM_PATH.
if (LBANN_SB_FWD_AWS_OFI_RCCL_RCCL_PATH)
  set(_aws_ofi_rccl_rccl_path_opt
    "--with-rccl=${LBANN_SB_FWD_AWS_OFI_RCCL_RCCL_PATH}")
elseif (LBANN_SB_BUILD_RCCL)
  ExternalProject_Get_property(RCCL INSTALL_DIR)
  set(_aws_ofi_rccl_rccl_path_opt
    "--with-rccl=${INSTALL_DIR}")
  set(_aws_ofi_rccl_rccl_dep
    "DEPENDS RCCL")
elseif (DEFINED ENV{ROCM_PATH})
  set(_aws_ofi_rccl_rccl_path_opt
    "--with-rccl=$ENV{ROCM_PATH}")
endif ()

# A user-provided path is preferred. Otherwise we try to
if (LBANN_SB_FWD_AWS_OFI_RCCL_LIBFABRIC_PATH)
  set(_aws_ofi_rccl_libfabric_path_opt
    "--with-libfabric=${LBANN_SB_FWD_AWS_OFI_RCCL_LIBFABRIC_PATH}")
else ()
  # First pkg-config, then probe the environment manually
  find_package(PkgConfig)
  if (PKG_CONFIG_FOUND)
    pkg_get_variable(_libfabric_dir libfabric prefix)
  endif ()
  if (NOT _libfabric_dir)
    find_program(_fi_info_exe fi_info)
    if (_fi_info_exe)
      get_filename_component(_fi_info_dir "${_fi_info_exe}" DIRECTORY)
      get_filename_component(_libfabric_dir "${_fi_info_dir}" DIRECTORY)
    endif ()
  endif ()
  if (_libfabric_dir)
    set(_aws_ofi_rccl_libfabric_path_opt
      "--with-libfabric=${_libfabric_dir}")
  endif ()
endif ()

# Finally we poke a bit at MPI. First we look for a user-provided
# path. Otherwise, we look for "MPICH_DIR" in the environment. This
# should be safe even if the Cray compiler is being used. If neither
# is found, we let the configure script do its thing and error out if
# it cannot sufficiently detect MPI.
if (LBANN_SB_FWD_AWS_OFI_RCCL_MPI_PATH)
  set(_aws_ofi_rccl_mpi_path_opt
    "--with-mpi=${LBANN_SB_FWD_AWS_OFI_RCCL_MPI_PATH}")
elseif (DEFINED ENV{MPICH_DIR})
  set(_aws_ofi_rccl_mpi_path_opt
    "--with-mpi=$ENV{MPICH_DIR}")
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

  # RCCL dependency, if needed.
  ${_aws_ofi_rccl_rccl_dep}

  CONFIGURE_COMMAND
  "${LBANN_SB_${PKG_NAME}_SOURCE_DIR}/configure"
  "--prefix=${LBANN_SB_${PKG_NAME}_PREFIX}"
  "CC=${LBANN_SB_${PKG_NAME}_C_COMPILER}"
  ${_aws_ofi_rccl_hip_path_opt}
  ${_aws_ofi_rccl_rccl_path_opt}
  ${_aws_ofi_rccl_libfabric_path_opt}
  ${_aws_ofi_rccl_mpi_path_opt}

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
