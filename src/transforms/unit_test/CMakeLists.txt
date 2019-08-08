set_full_path(_DIR_LBANN_CATCH2_TEST_FILES
  normalize_test.cpp
  sample_normalize_test.cpp
  scale_test.cpp
  transform_pipeline_test.cpp
  )

set(LBANN_CATCH2_TEST_FILES
  "${LBANN_CATCH2_TEST_FILES}" "${_DIR_LBANN_CATCH2_TEST_FILES}" PARENT_SCOPE)
