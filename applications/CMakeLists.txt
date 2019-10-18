
add_subdirectory(vision)
add_subdirectory(ICF)

# Install the relevant application directories
install(FILES README.md DESTINATION ${CMAKE_INSTALL_DATADIR}/applications)
install(DIRECTORY vision
  DESTINATION ${CMAKE_INSTALL_DATADIR}/applications
  FILES_MATCHING PATTERN "*.py")
