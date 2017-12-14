enable_language(C)
enable_language(CXX)

# Handle the clone mechanism. First URL
set(PROTOBUF_URL "https://github.com/google/protobuf.git"
  CACHE STRING "The URL from which to clone PROTOBUF")

# ... then the tag.
set(PROTOBUF_TAG "3.2.x"
  CACHE STRING "The git tag or hash to checkout for PROTOBUF")

# Where to install PROTOBUF
set(PROTOBUF_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}"
  CACHE PATH "The installation location of PROTOBUF.")

# The number of jobs to use when running 'make' for protobuf
set(PROTOBUF_MAKE_JOBS ""
  CACHE INTERNAL "The number of make jobs to run. Max by default.")

include(ExternalProject)
find_program(GNU_MAKE_PROGRAM NAMES make gmake nmake)
if (GNU_MAKE_PROGRAM)
  mark_as_advanced(GNU_MAKE_PROGRAM)
else ()
  message(FATAL_ERROR "GNU-compatible \'make\' program not found."
    "Protobuf requires \'make\'.")
endif ()

configure_file(protobuf_configure.sh.in
  ${CMAKE_CURRENT_BINARY_DIR}/tmp/protobuf_configure.sh @ONLY)

ExternalProject_Add(PROTOBUF
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}
  TMP_DIR ${CMAKE_CURRENT_BINARY_DIR}/tmp
  STAMP_DIR ${CMAKE_CURRENT_BINARY_DIR}/stamp
  GIT_REPOSITORY ${PROTOBUF_URL}
  GIT_TAG ${PROTOBUF_TAG}
  SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/src
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/build
  INSTALL_DIR ${CNPY_CMAKE_INSTALL_PREFIX}
  CONFIGURE_COMMAND ${CMAKE_CURRENT_BINARY_DIR}/tmp/protobuf_configure.sh
  BUILD_COMMAND ${GNU_MAKE_PROGRAM} -j${PROTOBUF_MAKE_JOBS}
  USES_TERMINAL_BUILD 1
  )

set(PROTOBUF_DIR ${PROTOBUF_INSTALL_PREFIX}
  CACHE INTERNAL "The install prefix of Protobuf.")
