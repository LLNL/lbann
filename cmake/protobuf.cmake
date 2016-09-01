include(ExternalProject)

# Protocol Buffers options
option(PROTOBUF_FORCE_BUILD "Protocol Buffers: force build?" ON)

# Check if Protocol Buffers is already installed
if(NOT PROTOBUF_FORCE_BUILD)
  find_package(Protobuf QUIET)
endif()

# Download and build Protocol Buffers if it is not found
if(NOT PROTOBUF_FOUND OR PROTOBUF_FORCE_BUILD)

  # Git repository URL and tag
  if(NOT DEFINED PROTOBUF_URL)
    set(PROTOBUF_URL "https://github.com/google/protobuf.git")
  endif()
  if(NOT DEFINED PROTOBUF_TAG)
     set(PROTOBUF_TAG "v3.0.0-beta-2")
  endif()
  message(STATUS "Will pull Protocol Buffers (tag ${PROTOBUF_TAG}) from ${PROTOBUF_URL}")

  # Download and build location
  set(PROTOBUF_SOURCE_DIR "${PROJECT_BINARY_DIR}/download/protobuf/source")
  set(PROTOBUF_BINARY_DIR "${PROJECT_BINARY_DIR}/download/protobuf/build")

  # Get Protocol Buffers from Git repository and build
  ExternalProject_Add(project_protobuf
    PREFIX         ${CMAKE_INSTALL_PREFIX}
    TMP_DIR        "${PROTOBUF_BINARY_DIR}/tmp"
    STAMP_DIR      "${PROTOBUF_BINARY_DIR}/stamp"
    GIT_REPOSITORY ${PROTOBUF_URL}
    GIT_TAG        ${PROTOBUF_TAG}
    SOURCE_DIR     ${PROTOBUF_SOURCE_DIR}
    CONFIGURE_COMMAND "${PROTOBUF_SOURCE_DIR}/autogen.sh && ${PROTOBUF_SOURCE_DIR}/configure --prefix=${CMAKE_INSTALL_PREFIX}"
    BINARY_DIR     ${PROTOBUF_BINARY_DIR}
#    BINARY_COMMAND "make"
    INSTALL_DIR    ${CMAKE_INSTALL_PREFIX}
#    INSTALL_COMMAND "make install"
  )

  # Get header files
  set(PROTOBUF_INCLUDE_DIRS "${CMAKE_INSTALL_PREFIX}/include")

  # Get library
  add_library(libprotobuf SHARED IMPORTED)
  set(PROTOBUF_LIBRARIES "${CMAKE_INSTALL_PREFIX}/lib/libprotobuf.so")
  set_property(TARGET libprotobuf PROPERTY IMPORTED_LOCATION ${PROTOBUF_LIBRARIES})

  # Get protoc compiler
  set(PROTOBUF_PROTOC_EXECUTABLE "${CMAKE_INSTALL_PREFIX}/bin/protoc")

  # LBANN has built Protocol Buffers
  set(LBANN_BUILT_PROTOBUF TRUE)

endif()

# LBANN has access to Protocol Buffers
if(PROTOBUF_FOUND OR LBANN_BUILT_PROTOBUF)
  set(LBANN_HAS_PROTOBUF TRUE)
endif()