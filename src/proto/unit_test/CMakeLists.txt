set_full_path(_DIR_LBANN_CATCH2_TEST_FILES
  parse_list_test.cpp
  parse_set_test.cpp
  trim_test.cpp
  )

set(LBANN_CATCH2_TEST_FILES
  "${LBANN_CATCH2_TEST_FILES}" "${_DIR_LBANN_CATCH2_TEST_FILES}" PARENT_SCOPE)
