set_full_path(THIS_DIR_SEQ_CATCH2_TEST_FILES
  data_reader_smiles_test.cpp
  )

set_full_path(THIS_DIR_MPI_CATCH2_TEST_FILES
  data_reader_smiles_fetch_datum_test.cpp
  data_reader_smiles_sample_list_test.cpp
  data_reader_HDF5_test.cpp
  data_reader_HDF5_sample_list_test.cpp
  )

set(LBANN_SEQ_CATCH2_TEST_FILES
  "${LBANN_SEQ_CATCH2_TEST_FILES}"
  "${THIS_DIR_SEQ_CATCH2_TEST_FILES}" PARENT_SCOPE)
set(LBANN_MPI_CATCH2_TEST_FILES
  "${LBANN_MPI_CATCH2_TEST_FILES}"
  "${THIS_DIR_MPI_CATCH2_TEST_FILES}" PARENT_SCOPE)
