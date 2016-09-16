# Try finding Doxygen
find_package(Doxygen QUIET)

if(DOXYGEN_FOUND)

  # Status message
  message(STATUS "Found Doxygen (version ${DOXYGEN_VERSION}): ${DOXYGEN_EXECUTABLE}")

  # Configure Doxygen configuration file
  configure_file(${PROJECT_SOURCE_DIR}/doc/Doxyfile.in
                 ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_DOCDIR}/Doxyfile @ONLY)

  # Generate documentation with Doxygen
  set(DOXYGEN_COMMAND ${DOXYGEN_EXECUTABLE} ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_DOCDIR}/Doxyfile)
  install(CODE "execute_process(COMMAND ${DOXYGEN_COMMAND} WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_DOCDIR})")

  # LBANN has access to Doxygen
  set(LBANN_HAS_DOXYGEN TRUE)

endif()
