set_full_path(_DIR_LBANN_CATCH2_TEST_FILES
  any_test.cpp
  beta_distribution_test.cpp
  factory_test.cpp
  image_test.cpp
  random_test.cpp
  type_erased_matrix_test.cpp
  )

set(LBANN_CATCH2_TEST_FILES
  "${LBANN_CATCH2_TEST_FILES}" "${_DIR_LBANN_CATCH2_TEST_FILES}" PARENT_SCOPE)
