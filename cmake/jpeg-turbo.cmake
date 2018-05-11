include(ExternalProject)

# Options
option(FORCE_JPEG_TURBO_BUILD "libjpeg-turbo: force build" OFF)
#set(JPEG_TURBO_MIN_VERSION "1.5.1")

# Check if libjpeg-turbo is already installed
if(NOT FORCE_JPEG_TURBO_BUILD)
  list(APPEND CMAKE_LIBRARY_PATH ${CMAKE_INSTALL_PREFIX}/lib)
  list(APPEND CMAKE_INCLUDE_PATH ${CMAKE_INSTALL_PREFIX}/include)
  find_package(JPEG_TURBO QUIET)
endif()

# Check if libjpeg-turbo has been found
if(JPEG_TURBO_FOUND AND NOT FORCE_JPEG_TURBO_BUILD)

  message(STATUS "Found libjpeg-turbo (version ${JPEG_TURBO_VERSION}): ${JPEG_TURBO_LIBRARIES}")

else()

  # Git repository URL and tag
  if(NOT JPEG_TURBO_URL)
    set(JPEG_TURBO_URL https://github.com/libjpeg-turbo/libjpeg-turbo)
  endif()
  if(NOT JPEG_TURBO_TAG)
     set(JPEG_TURBO_TAG "1.5.2")
  endif()
  message(STATUS "Will pull jpeg-turbo (tag ${JPEG_TURBO_TAG}) from ${JPEG_TURBO_URL}")

  # Download and build location
  set(JPEG_TURBO_SOURCE_DIR ${PROJECT_BINARY_DIR}/download/jpeg_turbo/source)
  set(JPEG_TURBO_BINARY_DIR ${PROJECT_BINARY_DIR}/download/jpeg_turbo/build)

  # header files
  set(JPEG_TURBO_INCLUDE_DIRS ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR})
  
  # library
  set(JPEG_TURBO_LIBRARIES ${CMAKE_INSTALL_PREFIX}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}jpeg${CMAKE_SHARED_LIBRARY_SUFFIX})

  # Get libjpeg-turbo from Git repository and build
  ExternalProject_Add(project_jpeg_turbo
    PREFIX            ${CMAKE_INSTALL_PREFIX}
    TMP_DIR           ${JPEG_TURBO_BINARY_DIR}/tmp
    STAMP_DIR         ${JPEG_TURBO_BINARY_DIR}/stamp
    GIT_REPOSITORY    ${JPEG_TURBO_URL}
    GIT_TAG           ${JPEG_TURBO_TAG}
    SOURCE_DIR        ${JPEG_TURBO_SOURCE_DIR}
    CONFIGURE_COMMAND pushd ${JPEG_TURBO_SOURCE_DIR} && test -f configure && echo "Skipping autorecnof" || autoreconf -fiv && ./configure --prefix=${CMAKE_INSTALL_PREFIX} CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER} && popd
    BINARY_DIR        ${JPEG_TURBO_BINARY_DIR}
    BUILD_COMMAND     pushd ${JPEG_TURBO_SOURCE_DIR} && test -f ${JPEG_TURBO_LIBRARIES} && echo "Skipping make" || ${CMAKE_MAKE_PROGRAM} -j${MAKE_NUM_PROCESSES} VERBOSE=${VERBOSE} && popd
    INSTALL_DIR       ${CMAKE_INSTALL_PREFIX}
    INSTALL_COMMAND   pushd ${JPEG_TURBO_SOURCE_DIR} && test -f ${JPEG_TURBO_LIBRARIES} && echo "Skipping make install" || ${CMAKE_MAKE_PROGRAM} install -j${MAKE_NUM_PROCESSES} VERBOSE=${VERBOSE} && popd
  )

  # LBANN has built libjpeg-turbo
  set(LBANN_BUILT_JPEG_TURBO TRUE)

endif()

# LBANN has access to libjpeg-turbo
if(JPEG_TURBO_FOUND OR LBANN_BUILT_JPEG_TURBO)
  include_directories(BEFORE ${JPEG_TURBO_INCLUDE_DIRS}) # Need to get ahead of system jpeg_turbo
  set(LBANN_HAS_JPEG_TURBO TRUE)
endif()
