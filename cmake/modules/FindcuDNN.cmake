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
#   CUDNN_FOUND
#   CUDNN_INCLUDE_PATH
#   CUDNN_LIBRARIES
#   CUDNN_VERSION
#
# Exports the following IMPORTED target
#
#   cuda::cudnn
#

find_path(CUDNN_INCLUDE_PATH cudnn.h
  HINTS ${CUDNN_DIR} $ENV{CUDNN_DIR} ${cuDNN_DIR} $ENV{cuDNN_DIR}
  ${CUDA_TOOLKIT_ROOT_DIR} $ENV{CUDA_TOOLKIT_ROOT_DIR}
  ${CUDA_SDK_ROOT_DIR}
  PATH_SUFFIXES include
  NO_DEFAULT_PATH
  DOC "Location of cudnn header."
  )
find_path(CUDNN_INCLUDE_PATH cudnn.h)

find_library(CUDNN_LIBRARY cudnn
  HINTS ${CUDNN_DIR} $ENV{CUDNN_DIR} ${cuDNN_DIR} $ENV{cuDNN_DIR}
  ${CUDA_TOOLKIT_ROOT_DIR} ${CUDA_SDK_ROOT_DIR}
  PATH_SUFFIXES lib64 lib
  NO_DEFAULT_PATH
  DOC "The cudnn library."
  )
find_library(CUDNN_LIBRARY cudnn)

# Get the version string
set(CUDNN_VERSION)
if (CUDNN_INCLUDE_PATH)
  set(_CUDNN_VERSION_SRC "
#include <stdio.h>
#include <cudnn_version.h>
int main() {
  printf(\"%d.%d.%d\", CUDNN_MAJOR, CUDNN_MINOR, CUDNN_PATCHLEVEL);
  return 0;
}
")

  file(
    WRITE
    "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/src.cxx"
    "${_CUDNN_VERSION_SRC}\n")

  try_run(
    _CUDNN_RUN_RESULT _CUDNN_COMPILE_RESULT
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/src.cxx
    CMAKE_FLAGS "-DINCLUDE_DIRECTORIES:STRING=${CUDNN_INCLUDE_PATH}"
    RUN_OUTPUT_VARIABLE CUDNN_VERSION
    COMPILE_OUTPUT_VARIABLE _CUDNN_COMPILE_OUTPUT)
endif ()

# Standard handling of the package arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(cuDNN
  DEFAULT_MSG CUDNN_VERSION CUDNN_LIBRARY CUDNN_INCLUDE_PATH)

if (NOT TARGET cuda::cudnn)

  add_library(cuda::cudnn INTERFACE IMPORTED)

  set_property(TARGET cuda::cudnn PROPERTY
    INTERFACE_INCLUDE_DIRECTORIES "${CUDNN_INCLUDE_PATH}")

  set_property(TARGET cuda::cudnn PROPERTY
    INTERFACE_LINK_LIBRARIES "${CUDNN_LIBRARY}")

endif (NOT TARGET cuda::cudnn)
