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
set_full_path(THIS_DIR_SEQ_CATCH2_TEST_FILES
  argument_parser_test.cpp
  beta_distribution_test.cpp
  cloneable_test.cpp
  dim_helpers_test.cpp
  environment_variable_test.cpp
  factory_test.cpp
  file_utils_test.cpp
  from_string_test.cpp
  hash_test.cpp
  output_helpers_test.cpp
  python_test.cpp
  random_test.cpp
  serialize_matrix_test.cpp
  statistics_test.cpp
  timer_test.cpp
  type_erased_matrix_test.cpp

  stubs/preset_env_accessor.hpp
  stubs/preset_env_accessor.cpp
  )

set_full_path(THIS_DIR_MPI_CATCH2_TEST_FILES
  random_fill_test.cpp
  rooted_archive_test.cpp
  serialize_distmatrix_test.cpp
  serialize_enum_test.cpp
  )

if (LBANN_HAS_HALF)
  list(APPEND THIS_DIR_SEQ_CATCH2_TEST_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/serialize_half_test.cpp)
endif (LBANN_HAS_HALF)

if (LBANN_HAS_FFTW)
  list(APPEND THIS_DIR_SEQ_CATCH2_TEST_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/fftw_test.cpp)
endif (LBANN_HAS_FFTW)

if (LBANN_HAS_OPENCV)
  list(APPEND THIS_DIR_SEQ_CATCH2_TEST_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/image_test.cpp)
endif (LBANN_HAS_OPENCV)

if (LBANN_HAS_ONNX)
  list(APPEND THIS_DIR_MPI_CATCH2_TEST_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/onnx_utils_test.cpp)
endif (LBANN_HAS_ONNX)

# This is actually CUDA-specific. rocFFT will get a new test when we
# port that over.
if (LBANN_HAS_CUDA)
  list(APPEND THIS_DIR_SEQ_CATCH2_TEST_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/cufft_test.cpp)
endif (LBANN_HAS_CUDA)

if (LBANN_HAS_DNN_LIB)
  list(APPEND THIS_DIR_SEQ_CATCH2_TEST_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/dnn_lib_test.cpp)
endif (LBANN_HAS_DNN_LIB)

if (LBANN_HAS_ONEDNN)
  message(STATUS "Adding oneDNN tests.")
  list(APPEND THIS_DIR_SEQ_CATCH2_TEST_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/onednn_infrastructure_test.cpp)
endif ()

set(LBANN_SEQ_CATCH2_TEST_FILES
  "${LBANN_SEQ_CATCH2_TEST_FILES}"
  "${THIS_DIR_SEQ_CATCH2_TEST_FILES}" PARENT_SCOPE)
set(LBANN_MPI_CATCH2_TEST_FILES
  "${LBANN_MPI_CATCH2_TEST_FILES}"
  "${THIS_DIR_MPI_CATCH2_TEST_FILES}" PARENT_SCOPE)
