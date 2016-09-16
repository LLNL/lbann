# Try finding Doxygen
find_package(Doxygen QUIET)

if(DOXYGEN_FOUND)

  # Status message
  message(STATUS "Found Doxygen (version ${DOXYGEN_VERSION}): ${DOXYGEN_EXECUTABLE}")

  # Generate documentation
  configure_file(${PROJECT_SOURCE_DIR}/doc/Doxyfile.in
                 ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_DOCDIR}/Doxyfile @ONLY)
  add_custom_target(doc
      ${DOXYGEN_EXECUTABLE} ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_DOCDIR}/Doxyfile
      WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_DOCDIR}
      COMMENT "Generating API documentation with Doxygen" VERBATIM
  )

  # LBANN has access to Doxygen
  set(LBANN_HAS_DOXYGEN TRUE)

endif()
