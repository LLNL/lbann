#
# Directly include example codes here
#

#
# This function adds a sequential (nompi) test.
#
function ( add_nonmpi_ctest test_name )
  set(test_source "test_${test_name}.cpp")
  set(test_exe    "test_${test_name}")
  add_executable(${test_exe} ${test_source})
  include_directories(${CMAKE_CURRENT_SOURCE_DIR})
  include_link_tbinf(${text_exe})
#  target_link_libraries(${test_exe} gtest gtest_main)
  add_test( "${test_name}_nompi" ${test_exe})
endfunction()

#
# This function adds a mpi test.
#
function ( add_mpi_ctest test_name )
  set(test_source "lbann_${test_name}.cpp")
  set(test_exe    "lbann_${test_name}")
  add_executable(${test_exe} ${test_source})
  include_link_tbinf(${test_exe})
  include_link_mpi(${test_exe})
  include_directories(${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(${test_exe} lbann)
#  target_link_libraries(${test_exe} gtest)
  SET( num_procs 1)
  WHILE( 8 GREATER ${num_procs} )
    add_test( "${test_name}_${num_procs}" ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG}
      ${num_procs} ${MPIEXEC_PREFLAGS} "${CMAKE_CURRENT_BINARY_DIR}/${test_exe}")
    MATH( EXPR num_procs "${num_procs} * 2" ) #double num_procs
  ENDWHILE()
endfunction()


#
# Sequential Tests
#
#add_nonmpi_ctest( sequential )

#
# Parallel Tests
#
add_mpi_ctest( dnn_mnist )
add_mpi_ctest( dnn_multi_mnist )
add_mpi_ctest( dnn_imagenet )
add_mpi_ctest( comm_test )
add_mpi_ctest( quantizer_test ) 
add_mpi_ctest( cnn_mnist )
#add_mpi_ctest( dnn_nci )
add_mpi_ctest( quantizer_bm )
