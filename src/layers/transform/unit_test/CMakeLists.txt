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
if (LBANN_HAS_TENSOR_PERMUTE)
  set_full_path(THIS_DIR_SEQ_CATCH2_TEST_FILES
    tensor_dims_utils_test.cpp
  )
  if (LBANN_HAS_CUTENSOR)
    list(APPEND THIS_DIR_SEQ_CATCH2_TEST_FILES
      "${CMAKE_CURRENT_SOURCE_DIR}/cutensor_permute_test.cpp")
  elseif (LBANN_HAS_CUTT OR LBANN_HAS_HIPTT)
    list(APPEND THIS_DIR_SEQ_CATCH2_TEST_FILES
      "${CMAKE_CURRENT_SOURCE_DIR}/cutt_permute_test.cpp")
  endif ()

  set_full_path(THIS_DIR_MPI_CATCH2_TEST_FILES
    permute_layer_test.cpp
  )
endif ()

set(LBANN_SEQ_CATCH2_TEST_FILES
  "${LBANN_SEQ_CATCH2_TEST_FILES}"
  "${THIS_DIR_SEQ_CATCH2_TEST_FILES}" PARENT_SCOPE)

set(LBANN_MPI_CATCH2_TEST_FILES
  "${LBANN_MPI_CATCH2_TEST_FILES}"
  "${THIS_DIR_MPI_CATCH2_TEST_FILES}" PARENT_SCOPE)
