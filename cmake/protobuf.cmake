include(ExternalProject)

# Options
option(FORCE_PROTOBUF_BUILD "Protocol Buffers: force build" OFF)

# Check if Protocol Buffers is already installed
if(NOT FORCE_PROTOBUF_BUILD)
  find_package(Protobuf QUIET)
endif()

# Check if Protocol Buffers has been found
if(PROTOBUF_FOUND AND NOT FORCE_PROTOBUF_BUILD)

  message(STATUS "Found Protocol Buffers: ${PROTOBUF_INCLUDE_DIRS} ${PROTOBUF_LIBRARIES}")

else()

  # Git repository URL and tag
  if(NOT PROTOBUF_URL)
    set(PROTOBUF_URL https://github.com/google/protobuf.git)
  endif()
  if(NOT PROTOBUF_TAG)
     set(PROTOBUF_TAG "v3.0.2")
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
    CONFIGURE_COMMAND pushd ${PROTOBUF_SOURCE_DIR} && ./autogen.sh && ./configure --prefix=${CMAKE_INSTALL_PREFIX} && popd
    BINARY_DIR        ${PROTOBUF_BINARY_DIR}
    BUILD_COMMAND     pushd ${PROTOBUF_SOURCE_DIR} && ${CMAKE_MAKE_PROGRAM} -j${MAKE_NUM_PROCESSES} VERBOSE=${VERBOSE} && popd
    INSTALL_DIR       ${CMAKE_INSTALL_PREFIX}
    INSTALL_COMMAND   pushd ${PROTOBUF_SOURCE_DIR} && ${CMAKE_MAKE_PROGRAM} install -j${MAKE_NUM_PROCESSES} VERBOSE=${VERBOSE} && popd
  )

  # Get header files
  set(PROTOBUF_INCLUDE_DIRS ${CMAKE_INSTALL_PREFIX}/include)
  
  # Get library
  set(PROTOBUF_LIBRARIES ${CMAKE_INSTALL_PREFIX}/lib/libprotobuf.so)

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