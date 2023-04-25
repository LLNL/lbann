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
  batchwise_reduce_sum.cpp
  bernoulli.cpp
  categorical_random.cpp
  concatenate.cpp
  constant.cpp
  crop.cpp
  cross_grid_sum_slice.cpp
  cross_grid_sum.cpp
  discrete_random.cpp
  dummy.cpp
  evaluation.cpp
  gather.cpp
  gaussian.cpp
  hadamard.cpp
  identity_zero.cpp
  in_top_k.cpp
  multidim_reduction.cpp
  pooling.cpp
  reduction.cpp
  reshape.cpp
  scatter.cpp
  slice.cpp
  sort.cpp
  split.cpp
  stop_gradient.cpp
  sum.cpp
  tessellate.cpp
  uniform.cpp
  unpooling.cpp
  upsample.cpp
  weighted_sum.cpp
  weights.cpp

  transform_builders.cpp
)

if (LBANN_HAS_GPU)
  if (LBANN_HAS_TENSOR_PERMUTE)
    list(APPEND THIS_DIR_SOURCES
      "${CMAKE_CURRENT_SOURCE_DIR}/permute/permuteimpl.hpp"
      "${CMAKE_CURRENT_SOURCE_DIR}/permute.cpp"
    )
    if (LBANN_HAS_CUTENSOR)
      list(APPEND THIS_DIR_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/permute/cutensor_permuteimpl.hpp"
      )
    elseif (LBANN_HAS_CUTT OR LBANN_HAS_HIPTT)
      list(APPEND THIS_DIR_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/permute/cutt_permuteimpl.hpp"
      )
    endif ()
  endif (LBANN_HAS_TENSOR_PERMUTE)

  # Add the CUDA source files for this directory
  set_full_path(THIS_DIR_CU_SOURCES
    concatenate.cu
    crop.cu
    gather.cu
    in_top_k.cu
    sort.cu
    scatter.cu
    slice.cu
    tessellate.cu
    split.cu
    sum.cu
    )
endif ()

# Add the subdirectories
add_subdirectory(cereal_registration)

if (LBANN_HAS_DISTCONV AND LBANN_HAS_NVSHMEM)
  # ADD the Distconv layer  subdirectory sources
  add_subdirectory(distconv)
endif()

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
set(GPU_SOURCES "${GPU_SOURCES}" "${THIS_DIR_CU_SOURCES}" PARENT_SCOPE)
