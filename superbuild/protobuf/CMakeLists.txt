enable_language(C)
enable_language(CXX)

# Option to use different compilers to, e.g., build a host version on
# a cross-compilation system (e.g., theta). You must specify
#
# HOST_PROTOBUF_C_COMPILER
# HOST_PROTOBUF_CXX_COMPILER
#
# and may optionally specify
#
# HOST_PROTOBUF_CMAKE_INSTALL_PREFIX (def: CMAKE_INSTALL_PREFIX/host-protobuf)
# HOST_PROTOBUF_CMAKE_BUILD_TYPE (def: CMAKE_BUILD_TYPE)
# HOST_PROTOBUF_CMAKE_C_FLAGS (def: <empty>)
# HOST_PROTOBUF_CMAKE_CXX_FLAGS (def: <empty>)
#
option(LBANN_SB_BUILD_PROTOBUF_HOST_VERSION
  "Build a protobuf that will run on the login/host node."
  OFF)

# Handle the clone mechanism. First URL
option(PROTOBUF_CLONE_VIA_SSH
  "Clone PROTOBUF using SSH instead of HTTPS" ${LBANN_SB_CLONE_VIA_SSH})

if (PROTOBUF_CLONE_VIA_SSH)
  set(PROTOBUF_URL git@github.com:protocolbuffers/protobuf.git
    CACHE STRING "The URL from which to clone PROTOBUF")
else ()
  set(PROTOBUF_URL "https://github.com/protocolbuffers/protobuf.git"
    CACHE STRING "The URL from which to clone PROTOBUF")
endif ()

# ... then the tag.
set(PROTOBUF_TAG "v3.10.1"
  CACHE STRING "The git tag or hash to checkout for PROTOBUF")

# Where to install PROTOBUF
set(PROTOBUF_CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}"
  CACHE PATH "The installation location of PROTOBUF.")

set(PROTOBUF_CMAKE_BUILD_TYPE "${CMAKE_BUILD_TYPE}"
  CACHE STRING "The build type for PROTOBUF.")

if (PROTOBUF_CUSTOM_SOURCE_DIR)
  set(PROTOBUF_SOURCE_DIR "${PROTOBUF_CUSTOM_SOURCE_DIR}")
  set(PROTOBUF_URL "")
  set(PROTOBUF_TAG "")
  set(_GIT_REPOSITORY_TAG)
  set(_GIT_TAG_TAG)
  message(STATUS "Using PROTOBUF source in: ${PROTOBUF_SOURCE_DIR}")
else ()
  set(PROTOBUF_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/src")
  set(_GIT_REPOSITORY_TAG "GIT_REPOSITORY")
  set(_GIT_TAG_TAG "GIT_TAG")
endif ()

include(ExternalProject)
ExternalProject_Add(PROTOBUF
  PREFIX "${CMAKE_CURRENT_BINARY_DIR}"
  TMP_DIR "${CMAKE_CURRENT_BINARY_DIR}/tmp"
  STAMP_DIR "${CMAKE_CURRENT_BINARY_DIR}/stamp"
  ${_GIT_REPOSITORY_TAG} ${PROTOBUF_URL}
  ${_GIT_TAG_TAG} ${PROTOBUF_TAG}
  SOURCE_DIR ${PROTOBUF_SOURCE_DIR}
  SOURCE_SUBDIR cmake
  BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/build"
  INSTALL_DIR "${PROTOBUF_CMAKE_INSTALL_PREFIX}"
  STEP_TARGETS download
  USES_TERMINAL_BUILD 1
  LOG_DOWNLOAD 1
  LOG_UPDATE 1
  LOG_CONFIGURE 1
  LOG_BUILD 1
  LOG_INSTALL 1
  LOG_TEST 1
  CMAKE_ARGS
  -G${CMAKE_GENERATOR}
  -DCMAKE_INSTALL_PREFIX=${PROTOBUF_CMAKE_INSTALL_PREFIX}
  -DCMAKE_BUILD_TYPE=${PROTOBUF_CMAKE_BUILD_TYPE}
  -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
  -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
  -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
  -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
  -DCMAKE_MACOSX_RPATH=ON
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON
  -Dprotobuf_BUILD_TESTS=OFF
  )
set(PROTOBUF_DIR ${PROTOBUF_CMAKE_INSTALL_PREFIX}
  CACHE INTERNAL "The install prefix of Protobuf.")

# Build the host-compatible version if necessary
if (LBANN_SB_BUILD_PROTOBUF_HOST_VERSION)
  set(HOST_PROTOBUF_CMAKE_INSTALL_PREFIX
    "${CMAKE_INSTALL_PREFIX}/host-protobuf"
    CACHE PATH
    "The installation location of host-compatible PROTOBUF.")

  set(HOST_PROTOBUF_CMAKE_BUILD_TYPE
    ${CMAKE_BUILD_TYPE}
    CACHE STRING
    "The build type for the host-compatible protobuf.")

  if (NOT HOST_PROTOBUF_CMAKE_C_COMPILER)
    message(FATAL_ERROR
      "Requested host protobuf build but did not specify a compiler. "
      "Please specify HOST_PROTOBUF_CMAKE_C_COMPILER and try again.")
  endif ()
  if (NOT HOST_PROTOBUF_CMAKE_CXX_COMPILER)
    message(FATAL_ERROR
      "Requested host protobuf build but did not specify a compiler. "
      "Please specify HOST_PROTOBUF_CMAKE_CXX_COMPILER and try again.")
  endif ()

  ExternalProject_Get_Property(PROTOBUF SOURCE_DIR)
  set(__host_protobuf_source_dir "${SOURCE_DIR}")
  set(SOURCE_DIR)

  ExternalProject_Add(HOST-PROTOBUF
    DEPENDS PROTOBUF-download
    PREFIX "${CMAKE_CURRENT_BINARY_DIR}"
    TMP_DIR "${CMAKE_CURRENT_BINARY_DIR}/tmp"
    STAMP_DIR "${CMAKE_CURRENT_BINARY_DIR}/stamp"
    DOWNLOAD_COMMAND ""
    SOURCE_DIR "${__host_protobuf_source_dir}"
    SOURCE_SUBDIR cmake
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/host-build"
    INSTALL_DIR "${HOST_PROTOBUF_CMAKE_INSTALL_PREFIX}"
    USES_TERMINAL_BUILD 1
    LOG_DOWNLOAD 1
    LOG_UPDATE 1
    LOG_CONFIGURE 1
    LOG_BUILD 1
    LOG_INSTALL 1
    LOG_TEST 1
    CMAKE_ARGS
    -DCMAKE_C_COMPILER=${HOST_PROTOBUF_CMAKE_C_COMPILER}
    -DCMAKE_CXX_COMPILER=${HOST_PROTOBUF_CMAKE_CXX_COMPILER}
    -DCMAKE_C_FLAGS=${HOST_PROTOBUF_CMAKE_C_FLAGS}
    -DCMAKE_CXX_FLAGS=${HOST_PROTOBUF_CMAKE_CXX_FLAGS}
    -DCMAKE_INSTALL_PREFIX=${HOST_PROTOBUF_CMAKE_INSTALL_PREFIX}
    -DCMAKE_BUILD_TYPE=${PROTOBUF_CMAKE_BUILD_TYPE}
    -DCMAKE_MACOSX_RPATH=ON
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    -Dprotobuf_BUILD_TESTS=OFF
    )

  set(HOST_PROTOBUF_DIR "${HOST_PROTOBUF_CMAKE_INSTALL_PREFIX}")
  set(HOST_PROTOBUF_protoc_EXE
    "${HOST_PROTOBUF_CMAKE_INSTALL_PREFIX}/bin/protoc"
    CACHE INTERNAL
    "Path to the host-compatible protoc compiler.")
endif ()
