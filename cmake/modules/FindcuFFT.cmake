################################################################################
## Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
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

#[=============[.rst
FindcuFFT
===============

Finds the cuFFT implementation that ships with the found version of
CUDA.

The following variables will be defined::

  cuFFT_FOUND         - True if the system has a cuFFT implementation.
  cuFFT_LIBRARIES     - Libraries for linking to cuFFT.
  cuFFT_INCLUDE_DIRS  - Include directories for cuFFT.

In addition, the :prop_tgt:`IMPORTED` target ``cuda::cufft`` will
be created.

#]=============]

find_path(CUFFT_INCLUDE_PATH cufft.h
  HINTS ${CUFFT_DIR} $ENV{CUFFT_DIR} ${cuFFT_DIR} $ENV{cuFFT_DIR}
  ${CUDA_TOOLKIT_ROOT_DIR} $ENV{CUDA_TOOLKIT_ROOT_DIR}
  ${CUDA_SDK_ROOT_DIR}
  PATH_SUFFIXES include
  NO_DEFAULT_PATH
  DOC "Location of cufft header."
  )
find_path(CUFFT_INCLUDE_PATH cufft.h)

find_library(CUFFT_LIBRARY cufft
  HINTS ${CUFFT_DIR} $ENV{CUFFT_DIR} ${cuFFT_DIR} $ENV{cuFFT_DIR}
  ${CUDA_TOOLKIT_ROOT_DIR} ${CUDA_SDK_ROOT_DIR}
  PATH_SUFFIXES lib64 lib
  NO_DEFAULT_PATH
  DOC "The cufft library."
  )
find_library(CUFFT_LIBRARY cufft)

# Standard handling of the package arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(cuFFT
  DEFAULT_MSG CUFFT_LIBRARY CUFFT_INCLUDE_PATH)

if (NOT TARGET cuda::cufft)

  add_library(cuda::cufft INTERFACE IMPORTED)

  set_property(TARGET cuda::cufft PROPERTY
    INTERFACE_INCLUDE_DIRECTORIES "${CUFFT_INCLUDE_PATH}")

  set_property(TARGET cuda::cufft PROPERTY
    INTERFACE_LINK_LIBRARIES "${CUFFT_LIBRARY}")

endif (NOT TARGET cuda::cufft)
