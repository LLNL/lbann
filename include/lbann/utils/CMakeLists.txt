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
# Add the headers for this directory
set_full_path(THIS_DIR_HEADERS
  amp.hpp
  any.hpp
  argument_parser.hpp
  beta.hpp
  cloneable.hpp
  commify.hpp
  compiler_control.hpp
  conduit_extensions.hpp
  dataset.hpp
  describable.hpp
  description.hpp
  dim_helpers.hpp
  dnn_enums.hpp
  entrywise_operator.hpp
  enum_iterator.hpp
  environment_variable.hpp
  exception.hpp
  factory.hpp
  factory_error_policies.hpp
  file_utils.hpp
  from_string.hpp
  glob.hpp
  graph.hpp
  hash.hpp
  hydrogen_utils.hpp
  im2col.hpp
  jag_utils.hpp
  lbann_library.hpp
  make_abstract.hpp
  memory.hpp
  mild_exception.hpp
  number_theory.hpp
  numerical_traits.hpp
  nvshmem.hpp
  omp_diagnostics.hpp
  omp_pragma.hpp
  onnx_utils.hpp
  options.hpp
  peek_map.hpp
  print_helpers.hpp
  profiling.hpp
  protobuf.hpp
  protobuf_serializable.hpp
  protobuf_utils.hpp
  python.hpp
  random.hpp
  random_number_generators.hpp
  serialize.hpp
  stack_trace.hpp
  statistics.hpp
  summary.hpp
  summary_impl.hpp
  sync_info_helpers.hpp
  system_info.hpp
  tensor.hpp
  tensor_impl.hpp
  timer.hpp
  trainer_file_utils.hpp
  type_erased_matrix.hpp
  typename.hpp
  vector_dynamic_cast.hpp
  vectorwrapbuf.hpp
  visitor_hooks.hpp
)
if (LBANN_HAS_FFTW)
  list(APPEND THIS_DIR_HEADERS
    "${CMAKE_CURRENT_SOURCE_DIR}/fft_common.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fftw_wrapper.hpp")
  if (LBANN_HAS_CUDA)
    list(APPEND THIS_DIR_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/cufft_wrapper.hpp")
  endif ()
endif (LBANN_HAS_FFTW)

if (LBANN_HAS_HALF)
  list(APPEND THIS_DIR_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/serialize.hpp")
endif (LBANN_HAS_HALF)

if (LBANN_HAS_DISTCONV)
  list(APPEND THIS_DIR_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/distconv.hpp")
endif ()

if (LBANN_HAS_OPENCV)
  list(APPEND THIS_DIR_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/opencv.hpp")
  list(APPEND THIS_DIR_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/image.hpp")
endif ()

if (LBANN_HAS_CNPY)
  list(APPEND THIS_DIR_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/cnpy_utils.hpp")
endif ()

# Add the subdirectories
add_subdirectory(threads)
add_subdirectory(impl)

# Propagate the files up the tree
set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
