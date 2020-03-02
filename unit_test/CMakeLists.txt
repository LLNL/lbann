# Add the unit testing utilities library
add_subdirectory(utilities)

# Add the sequential test main() function
add_executable(seq-catch-tests
  SequentialCatchMain.cpp "${LBANN_SEQ_CATCH2_TEST_FILES}")
target_link_libraries(seq-catch-tests
  PRIVATE unit_test_utilities lbann Catch2::Catch2)

catch_discover_tests(seq-catch-tests)

# There's an example MPI test
add_subdirectory(example)

# Add the parallel test main() function
add_executable(mpi-catch-tests
  MPICatchMain.cpp "${LBANN_MPI_CATCH2_TEST_FILES}")
target_link_libraries(mpi-catch-tests
  PRIVATE unit_test_utilities lbann Catch2::Catch2)

# TODO: Some "magical" way to automatically run tests if a parallel
# environment is detected at CTest time
