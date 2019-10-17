
# Install the relevant prototext
install(FILES README.md DESTINATION ${CMAKE_INSTALL_DATADIR}/model_zoo)
install(DIRECTORY data_readers models optimizers tests vision
  DESTINATION ${CMAKE_INSTALL_DATADIR}/model_zoo)
