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

# Exports the following variables
#
#   cuTENSOR_FOUND
#   cuTENSOR_INCLUDE_PATH
#   cuTENSOR_LIBRARIES
#   cuTENSOR_VERSION
#
# Exports the following IMPORTED target
#
#   cuTENSOR::cuTENSOR
#

# cuTENSOR relies on CUDA toolkit. LBANN requires > 11.
find_package(CUDAToolkit 11.0.0 REQUIRED)

find_path(cuTENSOR_INCLUDE_PATH cutensor.h
  HINTS
  ${cuTENSOR_DIR} $ENV{cuTENSOR_DIR}
  ${CUDA_TOOLKIT_ROOT_DIR} $ENV{CUDA_TOOLKIT_ROOT_DIR}
  ${CUDA_SDK_ROOT_DIR}
  PATH_SUFFIXES include
  NO_DEFAULT_PATH
  DOC "Location of cuTENSOR header.")
find_path(cuTENSOR_INCLUDE_PATH cutensor.h)

find_library(cuTENSOR_LIBRARY cutensor
  HINTS ${cuTENSOR_DIR} $ENV{cuTENSOR_DIR}
  ${CUDA_TOOLKIT_ROOT_DIR} $ENV{CUDA_TOOLKIT_ROOT_DIR}
  ${CUDA_SDK_ROOT_DIR}
  PATH_SUFFIXES
  lib64/${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR}
  lib64/${CUDAToolkit_VERSION_MAJOR}
  lib/${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR}
  lib/${CUDAToolkit_VERSION_MAJOR}
  lib64
  lib
  NO_DEFAULT_PATH
  DOC "The cuTENSOR library.")
find_library(cuTENSOR_LIBRARY cutensor)

# Get the version string
set(cuTENSOR_VERSION)
if (cuTENSOR_INCLUDE_PATH)

  set(_cuTENSOR_VERSION_SRC "
#include <stdio.h>
#include <cutensor.h>
int main() {
  printf(\"%d.%d.%d\", CUTENSOR_MAJOR, CUTENSOR_MINOR, CUTENSOR_PATCH);
  return 0;
}
")

  file(
    WRITE
    "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/src.cxx"
    "${_cuTENSOR_VERSION_SRC}\n")

  try_run(
    _cuTENSOR_RUN_RESULT _cuTENSOR_COMPILE_RESULT
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/src.cxx
    CMAKE_FLAGS "-DINCLUDE_DIRECTORIES:STRING=${cuTENSOR_INCLUDE_PATH}"
    LINK_LIBRARIES CUDA::cudart
    RUN_OUTPUT_VARIABLE cuTENSOR_VERSION
    COMPILE_OUTPUT_VARIABLE _cuTENSOR_COMPILE_OUTPUT)
endif ()

# Standard handling of the package arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(cuTENSOR
  DEFAULT_MSG cuTENSOR_VERSION cuTENSOR_LIBRARY cuTENSOR_INCLUDE_PATH)

if (NOT TARGET cuTENSOR::cuTENSOR)

  add_library(cuTENSOR::cuTENSOR INTERFACE IMPORTED)

  set_property(TARGET cuTENSOR::cuTENSOR PROPERTY
    INTERFACE_INCLUDE_DIRECTORIES "${cuTENSOR_INCLUDE_PATH}")

  set_property(TARGET cuTENSOR::cuTENSOR PROPERTY
    INTERFACE_LINK_LIBRARIES "${cuTENSOR_LIBRARY}")

endif (NOT TARGET cuTENSOR::cuTENSOR)
