################################################################################
## Copyright (c) 2014-2023, Lawrence Livermore National Security, LLC.
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
# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  argmax.cpp
  argmin.cpp
  channelwise_mean.cpp
  channelwise_softmax.cpp
  covariance.cpp
  dist_embedding.cpp
  external.cpp
  mini_batch_index.cpp
  mini_batch_size.cpp
  one_hot.cpp
  rowwise_weights_norms.cpp
  uniform_hash.cpp
  variance.cpp

  misc_builders.cpp
  )

if (LBANN_HAS_FFTW)
  list(APPEND THIS_DIR_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/dft_abs.cpp")
endif ()

if (LBANN_HAS_GPU)
  # Add the CUDA source files for this directory
  set_full_path(THIS_DIR_CU_SOURCES
    channelwise_mean.cu
    channelwise_softmax.cu
    covariance.cu
    dist_embedding.cu
    one_hot.cu
    rowwise_weights_norms.cu
    uniform_hash.cu
    variance.cu
    )
  if (LBANN_HAS_FFTW)
    list(APPEND THIS_DIR_CU_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/dft_abs.cu")
  endif ()
endif ()

# Add the subdirectories
add_subdirectory(cereal_registration)

if (LBANN_HAS_DISTCONV)
    add_subdirectory(distconv)
endif()

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
set(GPU_SOURCES "${GPU_SOURCES}" "${THIS_DIR_CU_SOURCES}" PARENT_SCOPE)
