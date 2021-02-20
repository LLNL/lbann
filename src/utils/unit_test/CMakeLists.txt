set_full_path(THIS_DIR_SEQ_CATCH2_TEST_FILES
  any_test.cpp
  argument_parser_test.cpp
  beta_distribution_test.cpp
  cloneable_test.cpp
  dim_helpers_test.cpp
  environment_variable_test.cpp
  factory_test.cpp
  file_utils_test.cpp
  from_string_test.cpp
  hash_test.cpp
  python_test.cpp
  random_test.cpp
  serialize_matrix_test.cpp
  type_erased_matrix_test.cpp

  stubs/preset_env_accessor.hpp
  stubs/preset_env_accessor.cpp
  )

set_full_path(THIS_DIR_MPI_CATCH2_TEST_FILES
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
