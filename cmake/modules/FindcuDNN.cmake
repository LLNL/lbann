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

# Standard handling of the package arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(cuDNN
  DEFAULT_MSG CUDNN_LIBRARY CUDNN_INCLUDE_PATH)

if (NOT TARGET cuda::cudnn)

  add_library(cuda::cudnn INTERFACE IMPORTED)

  set_property(TARGET cuda::cudnn PROPERTY
    INTERFACE_INCLUDE_DIRECTORIES "${CUDNN_INCLUDE_PATH}")

  set_property(TARGET cuda::cudnn PROPERTY
    INTERFACE_LINK_LIBRARIES "${CUDNN_LIBRARY}")

endif (NOT TARGET cuda::cudnn)
