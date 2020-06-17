# Use CLARA_URL to specify the location of the git repo. Use
# CLARA_TAG to specify the commit.

enable_language(CXX)

# Handle the clone mechanism. First URL
option(CLARA_CLONE_VIA_SSH
  "Clone CLARA using SSH instead of HTTPS" ${LBANN_SB_CLONE_VIA_SSH})

if (CLARA_CLONE_VIA_SSH)
  set(CLARA_URL git@github.com:catchorg/clara.git
    CACHE STRING "The URL from which to clone CLARA")
else ()
  set(CLARA_URL "https://github.com/catchorg/clara.git"
    CACHE STRING "The URL from which to clone CLARA")
endif ()

# ... then the tag.
set(CLARA_TAG "v1.1.5"
  CACHE STRING "The git tag or hash to checkout for CLARA")

# Where to install CLARA
set(CLARA_CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}"
  CACHE PATH "The installation location of CLARA.")

# The build type for CLARA
set(CLARA_CMAKE_BUILD_TYPE "${CMAKE_BUILD_TYPE}"
  CACHE STRING "The build type for CLARA.")

if (CLARA_CUSTOM_SOURCE_DIR)
  set(CLARA_SOURCE_DIR "${CLARA_CUSTOM_SOURCE_DIR}")
  set(CLARA_URL "")
  set(CLARA_TAG "")
  set(_GIT_REPOSITORY_TAG)
  set(_GIT_TAG_TAG)
  message(STATUS "Using CLARA source in: ${CLARA_SOURCE_DIR}")
else ()
  set(CLARA_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/src")
  set(_GIT_REPOSITORY_TAG "GIT_REPOSITORY")
  set(_GIT_TAG_TAG "GIT_TAG")
endif ()

set(CLARA_INSTALL_DIR ${CLARA_CMAKE_INSTALL_PREFIX}/include)

# Now add the external project
include(ExternalProject)
ExternalProject_Add(CLARA
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}
  TMP_DIR ${CMAKE_CURRENT_BINARY_DIR}/tmp
  STAMP_DIR ${CMAKE_CURRENT_BINARY_DIR}/stamp
  ${_GIT_REPOSITORY_TAG} ${CLARA_URL}
  ${_GIT_TAG_TAG} ${CLARA_TAG}
  SOURCE_DIR ${CLARA_SOURCE_DIR}
  BUILD_IN_SOURCE 1
  INSTALL_DIR ${CLARA_CMAKE_INSTALL_PREFIX}
  USES_TERMINAL_BUILD 1
  LOG_DOWNLOAD 1
  LOG_UPDATE 1
  LOG_CONFIGURE 1
  LOG_BUILD 1
  LOG_INSTALL 1
  LOG_TEST 1
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND
  ${CMAKE_COMMAND} -E make_directory ${CLARA_INSTALL_DIR}
  COMMAND
  ${CMAKE_COMMAND} -E copy
  ${CLARA_SOURCE_DIR}/single_include/clara.hpp
  ${CLARA_INSTALL_DIR}
  )

set(CLARA_DIR ${CLARA_CMAKE_INSTALL_PREFIX}
  CACHE INTERNAL "The install prefix of CLARA.")
