#Add a parallel test (MPI)
function(add_mpi_ctest TEST_NAME)

  # Add test executable
  set(TEST_SOURCE ${TEST_NAME}.cpp)
  set(TEST_EXE ${TEST_NAME})
  add_executable(${TEST_EXE} ${TEST_SOURCE})

  # Include header files and link libraries
  include_directories(${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(${TEST_EXE} ${MPI_CXX_LIBRARIES})
  target_link_libraries(${TEST_EXE} lbann)

  # ctest with 1, 2, and 4 MPI processes
  set(NUM_PROCS 1)
  while(8 GREATER ${NUM_PROCS})
    add_test("${TEST_NAME}_${NUM_PROCS}"
      ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${NUM_PROCS} ${MPIEXEC_PREFLAGS}
      ${CMAKE_CURRENT_BINARY_DIR}/${TEST_EXE})
    math(EXPR NUM_PROCS "${NUM_PROCS} * 2")
  endwhile()

endfunction()

# Parallel Tests
add_executable( lbann-bin lbann.cpp )
target_link_libraries(lbann-bin lbann )
set_target_properties(lbann-bin PROPERTIES OUTPUT_NAME lbann)


#this can be done simler - quick copy/paste hack //d hysom
add_executable( lbann-bin2 lbann2.cpp )
target_link_libraries(lbann-bin2 lbann )
set_target_properties(lbann-bin2 PROPERTIES OUTPUT_NAME lbann2)

# Install the binaries
install(
  TARGETS lbann-bin lbann-bin2
  EXPORT LBANNTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

# Install the relevant prototext
install(FILES README.md DESTINATION ${CMAKE_INSTALL_DATADIR}/model_zoo)
install(DIRECTORY data_readers models optimizers
  DESTINATION ${CMAKE_INSTALL_DATADIR}/model_zoo)
