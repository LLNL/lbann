add_executable( test_lbann_exception_tracing test_lbann_exception_tracing.cpp )
add_executable( test_sigint_tracing test_sigint_tracing.cpp ) 
add_executable( test_sigsev_tracing test_sigsev_tracing.cpp )

target_link_libraries( test_lbann_exception_tracing lbann )
target_link_libraries( test_sigint_tracing lbann )
target_link_libraries( test_sigsev_tracing lbann )
