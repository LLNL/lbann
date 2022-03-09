# This executable has to be given a different target name since the
# library target name is "lbann".
add_executable(lbann-bin lbann.cpp)
target_link_libraries(lbann-bin lbann)
set_target_properties(lbann-bin
  PROPERTIES
  OUTPUT_NAME lbann
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(LBANN_EXE_TGTS lbann-bin)

set(EXE_SRCS
  lbann_help.cpp
  lbann_gan.cpp
  lbann_cycgan.cpp
  lbann_aecycgan.cpp
  lbann_inf.cpp)
foreach (_src IN LISTS EXE_SRCS)
  get_filename_component(TGT_NAME "${_src}" NAME_WE)
  string(REPLACE "_" "-" TGT_NAME "${TGT_NAME}")

  add_executable(${TGT_NAME} "${_src}")
  target_link_libraries(${TGT_NAME} lbann)
  set_target_properties(${TGT_NAME}
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

  list(APPEND LBANN_EXE_TGTS ${TGT_NAME})
endforeach ()

# Install the binaries
install(
  TARGETS lbann-bin lbann-gan lbann-cycgan lbann-aecycgan
  lbann-help lbann-inf
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
