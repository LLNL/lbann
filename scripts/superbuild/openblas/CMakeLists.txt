################################################################################
## Copyright (c) 2014-2022, Lawrence Livermore National Security, LLC.
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
  NAME OpenBLAS
  LANGUAGES C Fortran
  GITHUB_URL xianyi/OpenBLAS.git
  GIT_TAG "v0.3.6")

option(OpenBLAS_USE_64_BIT_INDEX
  "Whether OpenBLAS should use 64-bit indices" OFF)

option(OpenBLAS_USE_OPENMP
  "Whether OpenBLAS should make use of OpenMP" ON)

set(OpenBLAS_MAX_MAKE_JOBS "4"
  CACHE STRING "Number of 'make' jobs allowed for OpenBLAS build.")

# Handle the options
if (OpenBLAS_USE_64_BIT_INDEX)
  set(_TMP_OpenBLAS_INTERFACE_COMMAND "INTERFACE64=1 SYMBOLSUFFIX=64")
else()
  set(_TMP_OpenBLAS_INTERFACE_COMMAND "INTERFACE64=0")
endif()

if (OpenBLAS_USE_OPENMP)
  set(_TMP_OpenBLAS_THREAD_COMMAND USE_OPENMP=1)
else()
  set(_TMP_OpenBLAS_THREAD_COMMAND USE_THREAD=0)
endif()

# Arch flags
#
# FIXME: No longer relevant, I'm guessing...
if(APPLE)
  # This is a hack but is a good default for modern Mac's
  set(_TMP_OpenBLAS_ARCH_COMMAND TARGET=SANDYBRIDGE)
endif()

set(OpenBLAS_INTERFACE_COMMAND "${_TMP_OpenBLAS_INTERFACE_COMMAND}"
  CACHE STRING
  "The command to be forwarded to OpenBLAS to control the index interface.")

set(OpenBLAS_THREAD_COMMAND "${_TMP_OpenBLAS_THREAD_COMMAND}"
  CACHE STRING
  "The command to be forwarded to OpenBLAS to control threading.")

set(OpenBLAS_ARCH_COMMAND "${_TMP_OpenBLAS_ARCH_COMMAND}"
  CACHE STRING
  "THe command to be forwarded to OpenBLAS to describe the CPU architecture.")

# If not using the Makefile generator for CMake, using
# CMAKE_MAKE_PROGRAM probably won't work here (in particular, ninja
# cannot process Makefiles). So we go looking for plain ol' "make"
# instead.
find_program(GNU_MAKE_PROGRAM make)

include (ExternalProject)
ExternalProject_Add(OpenBLAS
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}
  ${LBANN_SB_GIT_REPOSITORY_TAG} ${LBANN_SB_OpenBLAS_URL}
  ${LBANN_SB_GIT_TAG_TAG} ${LBANN_SB_OpenBLAS_TAG}
  TMP_DIR ${CMAKE_CURRENT_BINARY_DIR}/tmp
  STAMP_DIR ${CMAKE_CURRENT_BINARY_DIR}/stamp

  SOURCE_DIR ${LBANN_SB_OpenBLAS_SOURCE_DIR}
  INSTALL_DIR ${LBANN_SB_OpenBLAS_PREFIX}

  BUILD_IN_SOURCE 1

  USES_TERMINAL_BUILD 1
  LOG_DOWNLOAD 1
  LOG_UPDATE 1
  LOG_CONFIGURE 1
  LOG_BUILD 1
  LOG_INSTALL 1
  LOG_TEST 1

  CONFIGURE_COMMAND ""
  UPDATE_COMMAND ""

  BUILD_COMMAND
  ${GNU_MAKE_PROGRAM} -j${OpenBLAS_MAX_MAKE_JOBS}
  CC=${LBANN_SB_OpenBLAS_C_COMPILER}
  FC=${LBANN_SB_OpenBLAS_Fortran_COMPILER}
  ${OpenBLAS_THREAD_COMMAND}
  ${OpenBLAS_ARCH_COMMAND}
  ${OpenBLAS_INTERFACE_COMMAND}
  libs netlib shared
  INSTALL_COMMAND
  ${GNU_MAKE_PROGRAM} install PREFIX=${LBANN_SB_OpenBLAS_PREFIX}
)

set(OpenBLAS_DIR ${LBANN_SB_OpenBLAS_PREFIX}
  CACHE INTERNAL "The install prefix of OpenBLAS.")
