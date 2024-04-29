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

# The goal of this package is to enable *basic* builds of NCCL. Any
# configuration that would require modification to the provided
# makefile is considered out-of-scope and users needing such
# configuration should consider a standalone build rather than
# superbuilding NCCL.
#
# Note that the default NCCL makefile is rather rigid. It looks like
# one _can_ inject flags, but they have a high likelihood of being
# trampled by the makefile. E.g., the makefiles specifies optimization
# flags *after* the user injection point, so trying to modify the
# optimization level manually would be moot.

# Interprets the given variable as a boolean value and converts it to
# 1 (true) or 0 (false).
macro(bool_as_num VAR)
  if (${VAR})
    set(${VAR} 1)
  else ()
    set(${VAR} 0)
  endif ()
endmacro ()

lbann_sb_init_extern_pkg(
  NAME NCCL
  LANGUAGES C CXX # CUDA <- can't set explicitly; inferred from ${CUDA_HOME}
  GITHUB_URL NVIDIA/nccl
  GIT_TAG "master")

# User-facing options
lbann_sb_this_pkg_option(
  VERBOSE
  "Print build commands?"
  ON)
bool_as_num(LBANN_SB_FWD_NCCL_VERBOSE)

lbann_sb_this_pkg_option(
  KEEP
  "Keep intermediate files generated during compilation"
  OFF)
bool_as_num(LBANN_SB_FWD_NCCL_KEEP)

lbann_sb_this_pkg_option(
  ASAN
  "Build with address sanitizer enabled"
  OFF)
bool_as_num(LBANN_SB_FWD_NCCL_ASAN)

lbann_sb_this_pkg_option(
  TRACE
  "Build with tracing enabled"
  OFF)
bool_as_num(LBANN_SB_FWD_NCCL_TRACE)

# Debug build?
string(TOLOWER "${LBANN_SB_${PKG_NAME}_BUILD_TYPE}" _nccl_build_type)
if (_nccl_build_type STREQUAL "debug")
  set(_nccl_debug 1)
else ()
  set(_nccl_debug 0)
endif ()

# Prefer a user-specified CUDA path, then check CUDA_HOME
if (LBANN_SB_FWD_NCCL_CUDA_PATH)
  set(_nccl_cuda_path_opt
    "CUDA_HOME=${LBANN_SB_FWD_NCCL_CUDA_PATH}")
elseif (DEFINED ENV{CUDA_HOME})
  set(_nccl_cuda_path_opt
    "CUDA_HOME=$ENV{CUDA_HOME}")
else ()
  message(WARNING
    "You have enabled NCCL package, but CUDA_HOME "
    "is not available in your environment.")
endif ()

# Gencode control
if (LBANN_SB_FWD_NCCL_NVCC_GENCODE)
  set(_nccl_nvcc_gencode_opt
    "NVCC_GENCODE=${LBANN_SB_FWD_NCCL_NVCC_GENCODE}")
elseif (DEFINED $ENV{NVCC_GENCODE})
  set(_nccl_nvcc_gencode_opt
    "NVCC_GENCODE=$ENV{NVCC_GENCODE}")
else ()
  message(WARNING
    "You have enabled NCCL package, but you have not set "
    "the NVCC_GENCODE. This will build all gencodes supported "
    "by NCCL, which may increase the build time.")
endif ()

# The build system here is just a set of makefiles.
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

  GIT_SHALLOW 1

  BUILD_IN_SOURCE 1
  USES_TERMINAL_BUILD 1
  LOG_DOWNLOAD 1
  LOG_UPDATE 1
  LOG_CONFIGURE 1
  LOG_BUILD 1
  LOG_INSTALL 1
  LOG_TEST 1

  CONFIGURE_COMMAND ""

  BUILD_COMMAND
  ${GNU_MAKE_PROGRAM}
  src.build
  "PREFIX=${LBANN_SB_${PKG_NAME}_PREFIX}"
  "CC=${LBANN_SB_${PKG_NAME}_C_COMPILER}"
  "CXX=${LBANN_SB_${PKG_NAME}_CXX_COMPILER}"
  ${_nccl_cuda_path_opt}
  ${_nccl_nvcc_gencode_opt}
  "DEBUG=${_nccl_debug}"
  "VERBOSE=${LBANN_SB_FWD_NCCL_VERBOSE}"
  "KEEP=${LBANN_SB_FWD_NCCL_KEEP}"
  "ASAN=${LBANN_SB_FWD_NCCL_ASAN}"
  "TRACE=${LBANN_SB_FWD_NCCL_TRACE}"
  -j${${PKG_NAME}_MAX_MAKE_JOBS}

  INSTALL_COMMAND
  ${GNU_MAKE_PROGRAM}
  src.install
  "PREFIX=${LBANN_SB_${PKG_NAME}_PREFIX}"
  "CC=${LBANN_SB_${PKG_NAME}_C_COMPILER}"
  "CXX=${LBANN_SB_${PKG_NAME}_CXX_COMPILER}"
  ${_nccl_cuda_path_opt}
  ${_nccl_nvcc_gencode_opt}
  "DEBUG=${_nccl_debug}"
  "VERBOSE=${LBANN_SB_FWD_NCCL_VERBOSE}"
  "KEEP=${LBANN_SB_FWD_NCCL_KEEP}"
  "ASAN=${LBANN_SB_FWD_NCCL_ASAN}"
  "TRACE=${LBANN_SB_FWD_NCCL_TRACE}"
  -j${${PKG_NAME}_MAX_MAKE_JOBS}
)

set(${PKG_NAME}_DIR ${LBANN_SB_${PKG_NAME}_PREFIX}
  CACHE INTERNAL "The install prefix of ${PKG_NAME}.")
