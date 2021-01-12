
# Parallel Tests
add_executable( lbann-dr-bin test_bed_data_reader.cpp )
target_link_libraries(lbann-dr-bin lbann )
set_target_properties(lbann-dr-bin PROPERTIES OUTPUT_NAME test_bed_data_reader)

add_executable( lbann-bin lbann.cpp )
target_link_libraries(lbann-bin lbann )
set_target_properties(lbann-bin PROPERTIES OUTPUT_NAME lbann)

add_executable( lbann-help lbann_help.cpp )
target_link_libraries(lbann-help lbann )

#this can be done simler - quick copy/paste hack //d hysom
add_executable( lbann-gan-bin lbann_gan.cpp )
target_link_libraries(lbann-gan-bin lbann )
set_target_properties(lbann-gan-bin PROPERTIES OUTPUT_NAME lbann_gan)

add_executable( lbann-cycgan-bin lbann_cycgan.cpp )
target_link_libraries(lbann-cycgan-bin lbann )
set_target_properties(lbann-cycgan-bin PROPERTIES OUTPUT_NAME lbann_cycgan)

add_executable( lbann-aecycgan-bin lbann_aecycgan.cpp )
target_link_libraries(lbann-aecycgan-bin lbann )
set_target_properties(lbann-aecycgan-bin PROPERTIES OUTPUT_NAME lbann_aecycgan)

add_executable( lbann-inf-bin lbann_inf.cpp )
target_link_libraries(lbann-inf-bin lbann )
set_target_properties(lbann-inf-bin PROPERTIES OUTPUT_NAME lbann_inf)

# Install the binaries
install(
  TARGETS lbann-bin lbann-gan-bin lbann-cycgan-bin lbann-aecycgan-bin
  lbann-help lbann-inf-bin lbann-dr-bin
  EXPORT LBANNTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

# Install the relevant prototext
install(FILES README.md DESTINATION ${CMAKE_INSTALL_DATADIR}/model_zoo)
install(DIRECTORY data_readers models optimizers tests
  DESTINATION ${CMAKE_INSTALL_DATADIR}/model_zoo)
