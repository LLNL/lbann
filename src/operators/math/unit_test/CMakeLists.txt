set_full_path(THIS_DIR_MPI_CATCH2_TEST_FILES
  abs_test.cpp
  add_test.cpp
  clamp_test.cpp
  cos_test.cpp
  multiply_test.cpp
  sin_test.cpp
  subtract_test.cpp
  )

set(LBANN_MPI_CATCH2_TEST_FILES
  "${LBANN_MPI_CATCH2_TEST_FILES}"
  "${THIS_DIR_MPI_CATCH2_TEST_FILES}" PARENT_SCOPE)
