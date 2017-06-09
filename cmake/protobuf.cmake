include(ExternalProject)

# Options
option(FORCE_PROTOBUF_BUILD "Protocol Buffers: force build" OFF) # Many protobuf builds are version 2, which is too old
set(PROTOBUF_MIN_VERSION "3.0.0")

# Check if Protocol Buffers is already installed
if(NOT FORCE_PROTOBUF_BUILD)
  list(APPEND CMAKE_LIBRARY_PATH ${CMAKE_INSTALL_PREFIX}/lib)
  list(APPEND CMAKE_INCLUDE_PATH ${CMAKE_INSTALL_PREFIX}/include)
  find_package(Protobuf QUIET)
  execute_process(COMMAND ${PROTOBUF_PROTOC_EXECUTABLE} --version OUTPUT_VARIABLE PROTOBUF_PROTOC_VERSION)
  if ("${PROTOBUF_PROTOC_VERSION}" MATCHES "libprotoc ([0-9.]+)")
    set(PROTOBUF_VERSION "${CMAKE_MATCH_1}")
  endif()
endif()

# Check if Protocol Buffers has been found
if(PROTOBUF_FOUND AND NOT FORCE_PROTOBUF_BUILD AND (("${PROTOBUF_VERSION}" VERSION_EQUAL "${PROTOBUF_MIN_VERSION}") OR ("${PROTOBUF_VERSION}" VERSION_GREATER "${PROTOBUF_MIN_VERSION}")))

  message(STATUS "Found Protocol Buffers (version ${PROTOBUF_VERSION}): ${PROTOBUF_LIBRARIES}")

else()

  # Git repository URL and tag
  if(NOT PROTOBUF_URL)
    set(PROTOBUF_URL https://github.com/google/protobuf.git)
  endif()
  if(NOT PROTOBUF_TAG)
     set(PROTOBUF_TAG "3.2.x")
  endif()
  message(STATUS "Will pull Protocol Buffers (tag ${PROTOBUF_TAG}) from ${PROTOBUF_URL}")

  # Download and build location
  set(PROTOBUF_SOURCE_DIR ${PROJECT_BINARY_DIR}/download/protobuf/source)
  set(PROTOBUF_BINARY_DIR ${PROJECT_BINARY_DIR}/download/protobuf/build)

  # Get Protocol Buffers from Git repository and build
  ExternalProject_Add(project_protobuf
    PREFIX            ${CMAKE_INSTALL_PREFIX}
    TMP_DIR           ${PROTOBUF_BINARY_DIR}/tmp
    STAMP_DIR         ${PROTOBUF_BINARY_DIR}/stamp
    GIT_REPOSITORY    ${PROTOBUF_URL}
    GIT_TAG           ${PROTOBUF_TAG}
    SOURCE_DIR        ${PROTOBUF_SOURCE_DIR}
    CONFIGURE_COMMAND pushd ${PROTOBUF_SOURCE_DIR} && test -f configure && echo "Skipping autogen.sh" || ./autogen.sh  && test -f ${CMAKE_INSTALL_PREFIX}/bin/protoc && echo "Skipping configure" || ./configure --prefix=${CMAKE_INSTALL_PREFIX} CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER} CXX_FOR_BUILD=${CMAKE_CXX_COMPILER} && popd
    BINARY_DIR        ${PROTOBUF_BINARY_DIR}
    BUILD_COMMAND     pushd ${PROTOBUF_SOURCE_DIR} && test -f ${CMAKE_INSTALL_PREFIX}/bin/protoc && echo "Skipping make" || ${CMAKE_MAKE_PROGRAM} -j${MAKE_NUM_PROCESSES} VERBOSE=${VERBOSE} && popd
    INSTALL_DIR       ${CMAKE_INSTALL_PREFIX}
    INSTALL_COMMAND   pushd ${PROTOBUF_SOURCE_DIR} && test -f ${CMAKE_INSTALL_PREFIX}/bin/protoc && echo "Skipping make install" || ${CMAKE_MAKE_PROGRAM} install -j${MAKE_NUM_PROCESSES} VERBOSE=${VERBOSE} && popd
  )

  # Get header files
  set(PROTOBUF_INCLUDE_DIRS ${CMAKE_INSTALL_PREFIX}/include)
  
  # Get library
  set(PROTOBUF_LIBRARIES ${CMAKE_INSTALL_PREFIX}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}protobuf${CMAKE_SHARED_LIBRARY_SUFFIX})

  # Get protoc compiler
  set(PROTOBUF_PROTOC_EXECUTABLE ${CMAKE_INSTALL_PREFIX}/bin/protoc)

  # LBANN has built Protocol Buffers
  set(LBANN_BUILT_PROTOBUF TRUE)

endif()

# LBANN has access to Protocol Buffers
if(PROTOBUF_FOUND OR LBANN_BUILT_PROTOBUF)
  include_directories(BEFORE ${PROTOBUF_INCLUDE_DIRS}) # Need to get ahead of system protobuf
  set(LBANN_HAS_PROTOBUF TRUE)
endif()
