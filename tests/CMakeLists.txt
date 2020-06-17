add_executable( test_shuffled_indices test_shuffled_indices.cpp )
add_executable( test_mpi_err_handling test_mpi_err_handling.cpp )
target_link_libraries( test_shuffled_indices lbann )
target_link_libraries( test_mpi_err_handling lbann )
