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

# Sets the following variables
#
#   NVTX_FOUND
#   NVTX_LIBRARY
#
# Defines the following imported target:
#
#   cuda::nvtx
#

find_library(NVTX_LIBRARY nvToolsExt
  HINTS ${NVTX_DIR} $ENV{NVTX_DIR} ${CUDA_TOOLKIT_ROOT_DIR} ${CUDA_SDK_ROOT_DIR}
  PATH_SUFFIXES lib64
  DOC "The nvtx library."
  NO_DEFAULT_PATH)
find_library(NVTX_LIBRARY nvToolsExt)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NVTX
  DEFAULT_MSG NVTX_LIBRARY)

if (NOT TARGET cuda::nvtx)

  add_library(cuda::nvtx INTERFACE IMPORTED)

  set_property(TARGET cuda::nvtx PROPERTY
    INTERFACE_INCLUDE_DIRECTORIES "${CUDA_INCLUDE_DIRS}")

  set_property(TARGET cuda::nvtx PROPERTY
    INTERFACE_LINK_LIBRARIES "${NVTX_LIBRARY}")

endif (NOT TARGET cuda::nvtx)
