include(ExternalProject)

# CMake options
option(FORCE_CNPY_BUILD "cnpy: Force build" OFF)
if(NOT CNPY_LIBRARY_TYPE)
  set(CNPY_LIBRARY_TYPE SHARED)
endif()

# Try to use the find_package mechanism for CNPY
if (CNPY_DIR AND NOT FORCE_CNPY_BUILD)
  find_package(CNPY)
endif ()

# Download and build cnpy if required
if (NOT CNPY_FOUND)

  message(STATUS "No existing CNPY install found. Building from source.")
  
  # Git repository URL and tag
  if (NOT CNPY_URL)
    set(CNPY_URL https://github.com/rogersce/cnpy.git)
  endif ()
  if (NOT CNPY_TAG)
    set(CNPY_TAG "b39d58d3640f77c043bfe92591afeafdd82e78db")
  endif ()
  message(STATUS "Will pull CNPY (tag ${CNPY_TAG}) from ${CNPY_URL}")

  # Download and build location
  set(CNPY_SOURCE_DIR ${PROJECT_BINARY_DIR}/download/cnpy/source)
  set(CNPY_BINARY_DIR ${PROJECT_BINARY_DIR}/download/cnpy/build)

  # Build static library if required
  if (CNPY_LIBRARY_TYPE STREQUAL STATIC)
    set(CNPY_ENABLE_STATIC ON)
  else ()
    set(CNPY_ENABLE_STATIC OFF)
  endif ()

  # Get CNPY from Git repository
  ExternalProject_Add(project_CNPY
    PREFIX            ${CMAKE_INSTALL_PREFIX}
    TMP_DIR           ${CNPY_BINARY_DIR}/tmp
    STAMP_DIR         ${CNPY_BINARY_DIR}/stamp
    GIT_REPOSITORY    ${CNPY_URL}
    SOURCE_DIR        ${CNPY_SOURCE_DIR}
    BINARY_DIR        ${CNPY_BINARY_DIR}
    BUILD_COMMAND     ${CMAKE_MAKE_PROGRAM} VERBOSE=${VERBOSE}
    INSTALL_DIR       ${CMAKE_INSTALL_PREFIX}
    INSTALL_COMMAND   ${CMAKE_MAKE_PROGRAM} install VERBOSE=${VERBOSE}
    CMAKE_ARGS
      -D CMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
      -D CMAKE_INSTALL_MESSAGE=${CMAKE_INSTALL_MESSAGE}
      -D CMAKE_BUILD_TYPE=${CNPY_BUILD_TYPE}
      -D CMAKE_C_COMPILER=${CMAKE_C_COMPILER}
      -D CMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
      -D CMAKE_SKIP_BUILD_RPATH=${CMAKE_SKIP_BUILD_RPATH}
      -D CMAKE_BUILD_WITH_INSTALL_RPATH=${CMAKE_BUILD_WITH_INSTALL_RPATH}
      -D CMAKE_INSTALL_RPATH_USE_LINK_PATH=${CMAKE_INSTALL_RPATH_USE_LINK_PATH}
      -D CMAKE_INSTALL_RPATH=${CMAKE_INSTALL_RPATH}
      -D CMAKE_MACOSX_RPATH=${CMAKE_MACOSX_RPATH}
      -D ENABLE_STATIC=${CNPY_ENABLE_STATIC}
  )

  # Get install directory
  set(CNPY_DIR ${CMAKE_INSTALL_PREFIX})

  # LBANN has built CNPY
  set(LBANN_BUILT_CNPY TRUE)
  
  # Include header files
  set(CNPY_INCLUDE_DIRS ${CNPY_DIR}/include)

  # Get library
  if (CNPY_LIBRARY_TYPE STREQUAL STATIC)
    set(CNPY_LIBRARIES ${CNPY_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}cnpy${CMAKE_STATIC_LIBRARY_SUFFIX})
  else ()
    set(CNPY_LIBRARIES ${CNPY_DIR}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}cnpy${CMAKE_SHARED_LIBRARY_SUFFIX})
  endif ()

endif ()

# FIXME: This should not be done this way; rather,
# target_include_directories() should be used. Or some sort of
# interface target.
include_directories(${CNPY_INCLUDE_DIRS})

# LBANN has access to CNPY
set(LBANN_HAS_CNPY TRUE)
