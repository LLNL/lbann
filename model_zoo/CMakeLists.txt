# Parallel Tests
add_executable( lbann-bin lbann.cpp )
target_link_libraries(lbann-bin lbann )
set_target_properties(lbann-bin PROPERTIES OUTPUT_NAME lbann)


#this can be done simler - quick copy/paste hack //d hysom
add_executable( lbann-bin2 lbann2.cpp )
target_link_libraries(lbann-bin2 lbann )
set_target_properties(lbann-bin2 PROPERTIES OUTPUT_NAME lbann2)

add_executable( lbann-gan-bin lbann_gan.cpp )
target_link_libraries(lbann-gan-bin lbann )
set_target_properties(lbann-gan-bin PROPERTIES OUTPUT_NAME lbann_gan)
# Install the binaries
install(
  TARGETS lbann-bin lbann-bin2 lbann-gan-bin
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
