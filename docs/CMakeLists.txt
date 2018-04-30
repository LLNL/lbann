find_package(Doxygen)

if (DOXYGEN_FOUND)
  set(DOXYGEN_SOURCES
    Doxyfile.in
    callbacks.dox
    developer_guide.dox
    getting_started.dox
    layers.dox
    mainpage.dox
    metrics.dox
    objective_functions.dox
    optimizers.dox
    using_tensorboard.dox
    )

  # Setup Dot support
  if (DOXYGEN_DOT_FOUND)
    set(DOXYGEN_HAVE_DOT "YES")
    get_filename_component(DOXYGEN_DOT_PATH "${DOXYGEN_DOT_EXECUTABLE}"
      DIRECTORY)
  else ()
    set(DOXYGEN_HAVE_DOT "NO")
  endif()

  set(DOXYGEN_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/doxygen")

  # Setup the doxyfile
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in"
    "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile" @ONLY)

  # Apparently the Makefile generator doesn't do this properly
  add_custom_target(make_doxygen_dir
    COMMAND ${CMAKE_COMMAND} -E make_directory "${DOXYGEN_OUTPUT_DIR}"
    COMMENT "Creating the doxygen output directory."
    VERBATIM)

  # These are not just called "doc" so that we may one day support
  # many different in-source documentation platforms. This may be
  # overly optimistic.
  add_custom_target(doxygen
    COMMAND "${DOXYGEN_EXECUTABLE}" "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile"
    BYPRODUCTS "${DOXYGEN_OUTPUT_DIR}/html/index.html"
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
    SOURCES "${DOXYGEN_SOURCES}"
    COMMENT
    "Generating Doxygen documentation as ${DOXYGEN_OUTPUT_DIR}/html/index.html"
    VERBATIM)
  add_dependencies(doxygen make_doxygen_dir)

  add_custom_target(clean-doxygen
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${DOXYGEN_OUTPUT_DIR}
    COMMENT "Removing Doxygen documentation."
    VERBATIM)

  # Add shortcuts that will, in the distant and possibly fictitious
  # future, build all types of documentation (e.g. Doxygen, Sphinx,
  # the next great thing in code documentation, etc...)
  add_custom_target(doc)
  add_dependencies(doc doxygen)

  add_custom_target(clean-doc)
  add_dependencies(clean-doc clean-doxygen)
endif()
