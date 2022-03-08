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
# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  batchwise_reduce_sum.cpp
  batchwise_reduce_sum_builder.cpp
  bernoulli.cpp
  categorical_random.cpp
  concatenate.cpp
  concatenate_builder.cpp
  constant.cpp
  crop.cpp
  crop_builder.cpp
  cross_grid_sum_slice.cpp
  cross_grid_sum.cpp
  discrete_random.cpp
  dummy.cpp
  evaluation.cpp
  gather.cpp
  gather_builder.cpp
  gaussian.cpp
  hadamard.cpp
  in_top_k.cpp
  pooling.cpp
  reduction.cpp
  reshape.cpp
  scatter.cpp
  scatter_builder.cpp
  slice.cpp
  sort.cpp
  split.cpp
  stop_gradient.cpp
  sum.cpp
  tessellate.cpp
  uniform.cpp
  unpooling.cpp
  weighted_sum.cpp
  weights.cpp
  )

if (LBANN_HAS_GPU)
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

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
set(GPU_SOURCES "${GPU_SOURCES}" "${THIS_DIR_CU_SOURCES}" PARENT_SCOPE)
