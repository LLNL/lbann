set_full_path(THIS_DIR_MPI_CATCH2_TEST_FILES
  data_coordinator_HDF5_hrrl_public_api.cpp
  buffered_data_coordinator_test.cpp
  )

set(LBANN_MPI_CATCH2_TEST_FILES
  "${LBANN_MPI_CATCH2_TEST_FILES}"
  "${THIS_DIR_MPI_CATCH2_TEST_FILES}" PARENT_SCOPE)
