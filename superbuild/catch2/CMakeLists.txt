# Use CATCH2_URL to specify the location of the git repo. Use
# CATCH2_TAG to specify the commit.

enable_language(CXX)

# Handle the clone mechanism. First URL
option(CATCH2_CLONE_VIA_SSH
  "Clone CATCH2 using SSH instead of HTTPS" ${LBANN_SB_CLONE_VIA_SSH})

if (CATCH2_CLONE_VIA_SSH)
  set(CATCH2_URL git@github.com:catchorg/catch2.git
    CACHE STRING "The URL from which to clone CATCH2")
else ()
  set(CATCH2_URL "https://github.com/catchorg/catch2.git"
    CACHE STRING "The URL from which to clone CATCH2")
endif ()

# ... then the tag.
set(CATCH2_TAG "v2.9.2"
  CACHE STRING "The git tag or hash to checkout for CATCH2")

# Where to install CATCH2
set(CATCH2_CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}"
  CACHE PATH "The installation location of CATCH2.")

# The build type for CATCH2
set(CATCH2_CMAKE_BUILD_TYPE "${CMAKE_BUILD_TYPE}"
  CACHE STRING "The build type for CATCH2.")

if (CATCH2_CUSTOM_SOURCE_DIR)
  set(CATCH2_SOURCE_DIR "${CATCH2_CUSTOM_SOURCE_DIR}")
  set(CATCH2_URL "")
  set(CATCH2_TAG "")
  set(_GIT_REPOSITORY_TAG)
  set(_GIT_TAG_TAG)
  message(STATUS "Using CATCH2 source in: ${CATCH2_SOURCE_DIR}")
else ()
  set(CATCH2_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/src")
  set(_GIT_REPOSITORY_TAG "GIT_REPOSITORY")
  set(_GIT_TAG_TAG "GIT_TAG")
endif ()

# Now add the external project
include(ExternalProject)
ExternalProject_Add(CATCH2
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}
  TMP_DIR ${CMAKE_CURRENT_BINARY_DIR}/tmp
  STAMP_DIR ${CMAKE_CURRENT_BINARY_DIR}/stamp
  ${_GIT_REPOSITORY_TAG} ${CATCH2_URL}
  ${_GIT_TAG_TAG} ${CATCH2_TAG}
  SOURCE_DIR ${CATCH2_SOURCE_DIR}
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/build
  INSTALL_DIR ${CATCH2_CMAKE_INSTALL_PREFIX}
  USES_TERMINAL_BUILD 1
  LOG_DOWNLOAD 1
  LOG_UPDATE 1
  LOG_CONFIGURE 1
  LOG_BUILD 1
  LOG_INSTALL 1
  LOG_TEST 1
  CMAKE_ARGS
  -G${CMAKE_GENERATOR}
  -DCMAKE_INSTALL_PREFIX=${CATCH2_CMAKE_INSTALL_PREFIX}
  -DCMAKE_BUILD_TYPE=${CATCH2_CMAKE_BUILD_TYPE}
  -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
  -DCATCH_BUILD_TESTING=OFF
  -DCATCH_BUILDE_EXAMPLES=OFF
  -DCATCH_ENABLE_WERROR=OFF
  )

set(CATCH2_DIR ${CATCH2_CMAKE_INSTALL_PREFIX}
  CACHE INTERNAL "The install prefix of CATCH2.")
