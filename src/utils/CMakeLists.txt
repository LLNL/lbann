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
  argument_parser.cpp
  commify.cpp
  cudnn.cpp
  dataset.cpp
  description.cpp
  environment_variable.cpp
  exception.cpp
  file_utils.cpp
  graph.cpp
  im2col.cpp
  jag_common.cpp
  lbann_library.cpp
  miopen.cpp
  number_theory.cpp
  omp_diagnostics.cpp
  options.cpp
  output_helpers.cpp
  profiling.cpp
  protobuf.cpp
  protobuf_utils.cpp
  python.cpp
  random.cpp
  random_number_generators.cpp
  serialization.cpp
  stack_trace.cpp
  statistics.cpp
  summary.cpp
  system_info.cpp
  timer_map.cpp
  trainer_file_utils.cpp
  typename.cpp
  visitor_hooks.cpp
)

if (LBANN_HAS_CUDA)
  list(APPEND THIS_DIR_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/cufft_wrapper.cpp")

  # Add the CUDA source files for this directory
  set_full_path(THIS_DIR_CU_SOURCES
    cuda.cu
    nvshmem.cu
    im2col.cu
    )
endif ()

if (LBANN_HAS_ROCM)
  # Add the ROCM source files for this directory
  set_full_path(THIS_DIR_CU_SOURCES
    im2col.cu
    rocm.cpp
    )
endif ()

if (LBANN_HAS_ONEDNN)
  list(APPEND THIS_DIR_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/onednn.cpp")
endif ()

if (LBANN_HAS_DISTCONV)
  list(APPEND THIS_DIR_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/distconv.cpp")
endif ()

if (LBANN_HAS_OPENCV)
  list(APPEND THIS_DIR_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/image.cpp")
endif ()

if (LBANN_HAS_CNPY)
  list(APPEND THIS_DIR_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/cnpy_utils.cpp")
endif ()

add_subdirectory(serialize_matrices)
add_subdirectory(threads)

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
set(GPU_SOURCES "${GPU_SOURCES}" "${THIS_DIR_CU_SOURCES}" PARENT_SCOPE)
