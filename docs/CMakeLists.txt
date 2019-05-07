find_package(Doxygen)
find_package(Sphinx)

if (SPHINX_FOUND OR DOXYGEN_FOUND)
  add_custom_target(doc)
  add_custom_target(clean-doc)
endif ()

if (SPHINX_FOUND)
  find_package(Breathe)
  if (NOT BREATHE_FOUND)
    message(WARNING
      "Sphinx has been detected, but could not find the Breathe tool. "
      "This is currently required for building LBANN's sphinx documentation. "
      "Please install breathe (e.g., \"pip install breathe\") and reconfigure.")
  else ()
    set_full_path(SPHINX_SOURCES
      Makefile
      conf.py
      build_containers.rst
      building_lbann.rst
      build_llnl_idiosyncracies.rst
      build_spack_extra_config.rst
      build_with_cmake.rst
      build_with_superbuild.rst
      build_osx.rst
      publications.rst
      index.rst
      running_lbann.rst
      sphinx_requirements.txt
      )

    if (NOT SPHINX_BUILDER)
      set(SPHINX_BUILDER "html")
    endif ()
    set(SPHINX_OUTPUT_DIR
      "${CMAKE_CURRENT_BINARY_DIR}/sphinx/${SPHINX_BUILDER}")

    # Apparently the Makefile generator doesn't do this properly
    add_custom_target(make_sphinx_dir
      COMMAND ${CMAKE_COMMAND} -E make_directory "${SPHINX_OUTPUT_DIR}"
      COMMENT "Creating the sphinx output directory."
      VERBATIM)

    set(SPHINX_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
    set(SPHINX_BUILD_DIR "${SPHINX_OUTPUT_DIR}")

    add_custom_target(sphinx
      COMMAND "${SPHINX_EXECUTABLE}" "-b" "${SPHINX_BUILDER}"
      "${SPHINX_SOURCE_DIR}" "${SPHINX_BUILD_DIR}"
      BYPRODUCTS "${SPHINX_OUTPUT_DIR}/index.html"
      WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
      SOURCES "${SPHINX_SOURCES}"
      COMMENT
      "Generating Sphinx documentation as ${SPHINX_OUTPUT_DIR}/index.html"
      VERBATIM)
    add_dependencies(sphinx make_sphinx_dir)

    add_custom_target(clean-sphinx
      COMMAND ${CMAKE_COMMAND} -E remove_directory ${SPHINX_OUTPUT_DIR}
      COMMENT "Removing Sphinx documentation."
      VERBATIM)

    add_dependencies(doc sphinx)
    add_dependencies(clean-doc clean-sphinx)
  endif ()
endif ()

if (DOXYGEN_FOUND)
  set(DOXYGEN_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in
    ${CMAKE_SOURCE_DIR}/README.md
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
  add_dependencies(doc doxygen)
  add_dependencies(clean-doc clean-doxygen)
endif()
