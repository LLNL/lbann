set_full_path(THIS_DIR_SEQ_CATCH2_TEST_FILES
  any_test.cpp
  argument_parser_test.cpp
  beta_distribution_test.cpp
  environment_variable_test.cpp
  factory_test.cpp
  from_string_test.cpp
  hash_test.cpp
  image_test.cpp
  python_test.cpp
  random_test.cpp
  type_erased_matrix_test.cpp

  stubs/preset_env_accessor.hpp
  stubs/preset_env_accessor.cpp
  )

set(LBANN_SEQ_CATCH2_TEST_FILES
  "${LBANN_SEQ_CATCH2_TEST_FILES}"
  "${THIS_DIR_SEQ_CATCH2_TEST_FILES}" PARENT_SCOPE)
