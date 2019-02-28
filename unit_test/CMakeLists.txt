# Add the sequential test main() function
add_executable(seq-catch-tests
  SequentialCatchMain.cpp "${LBANN_CATCH2_TEST_FILES}")
target_link_libraries(seq-catch-tests PRIVATE lbann Catch2::Catch2)

catch_discover_tests(seq-catch-tests)

# Add the parallel test main() function -- TODO
