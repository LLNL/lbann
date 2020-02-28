set_full_path(THIS_DIR_SEQ_CATCH2_TEST_FILES
  normalize_test.cpp
  sample_normalize_test.cpp
  scale_test.cpp
  transform_pipeline_test.cpp
  )

set(LBANN_SEQ_CATCH2_TEST_FILES
  "${LBANN_SEQ_CATCH2_TEST_FILES}"
  "${THIS_DIR_SEQ_CATCH2_TEST_FILES}"
  PARENT_SCOPE)
