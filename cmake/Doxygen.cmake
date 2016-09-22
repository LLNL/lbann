# Try finding Doxygen
find_package(Doxygen QUIET)

if(DOXYGEN_FOUND)

  # Status message
  message(STATUS "Found Doxygen (version ${DOXYGEN_VERSION}): ${DOXYGEN_EXECUTABLE}")

  # Doxygen options
  if(NOT DOXYGEN_OUTPUT_DIR)
    set(DOXYGEN_OUTPUT_DIR ${PROJECT_INSTALL_PREFIX}/${CMAKE_INSTALL_DOCDIR})
  endif()

  # Configure Doxygen configuration file
  set(DOXYFILE ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_DOCDIR}/Doxyfile)
  configure_file(${PROJECT_SOURCE_DIR}/doc/Doxyfile.in
                 ${DOXYFILE} @ONLY)

  # Generate documentation
  add_custom_target(doc
    ${DOXYGEN_EXECUTABLE} ${DOXYFILE}
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_DOCDIR}
    COMMENT "Generating API documentation with Doxygen" VERBATIM
  )

  # LBANN has access to Doxygen
  set(LBANN_HAS_DOXYGEN TRUE)

endif()
