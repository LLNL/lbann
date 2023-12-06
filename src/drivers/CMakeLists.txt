# This executable has to be given a different target name since the
# library target name is "lbann".
add_executable(lbann-bin lbann.cpp)
target_link_libraries(lbann-bin lbann)
set_target_properties(lbann-bin
  PROPERTIES
  OUTPUT_NAME lbann)

add_executable(lbann-help lbann_help.cpp)
target_link_libraries(lbann-help lbann)

set_target_properties(lbann-bin lbann-help
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Install the binaries
install(
  TARGETS lbann-bin lbann-help
  EXPORT LBANNTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
