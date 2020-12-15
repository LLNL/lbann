set_full_path(THIS_DIR_SEQ_CATCH2_TEST_FILES
  test_adagrad.cpp
  test_adam.cpp
  test_hypergradient_adam.cpp
  test_rmsprop.cpp
  test_sgd.cpp
  )

set(LBANN_SEQ_CATCH2_TEST_FILES
  "${LBANN_SEQ_CATCH2_TEST_FILES}"
  "${THIS_DIR_SEQ_CATCH2_TEST_FILES}" PARENT_SCOPE)
