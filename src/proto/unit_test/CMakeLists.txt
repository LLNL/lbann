set_full_path(THIS_DIR_SEQ_CATCH2_TEST_FILES
  parse_list_test.cpp
  parse_set_test.cpp
  trim_test.cpp
  )

set(LBANN_SEQ_CATCH2_TEST_FILES
  "${LBANN_SEQ_CATCH2_TEST_FILES}"
  "${THIS_DIR_SEQ_CATCH2_TEST_FILES}"
  PARENT_SCOPE)
