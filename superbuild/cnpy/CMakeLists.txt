# Use CNPY_URL to specify the location of the git repo. Use CNPY_TAG
# to specify the commit.

enable_language(C)
enable_language(CXX)

# Match the CNPY defaults
option(CNPY_BUILD_SHARED_LIBS
  "Build the CNPY shared library" ${BUILD_SHARED_LIBS})

option(CNPY_ENABLE_STATIC "Enable the CNPY static linkage." ON)

# Handle the clone mechanism. First URL
option(CNPY_CLONE_VIA_SSH
  "Clone CNPY using SSH instead of HTTPS" ${LBANN_SB_CLONE_VIA_SSH})

if (CNPY_CLONE_VIA_SSH)
  set(CNPY_URL git@github.com:rogersce/cnpy.git
    CACHE STRING "The URL from which to clone CNPY")
else ()
  set(CNPY_URL "https://github.com/rogersce/cnpy.git"
    CACHE STRING "The URL from which to clone CNPY")
endif ()

# ... then the tag.
set(CNPY_TAG "4e8810b1a8637695171ed346ce68f6984e585ef4"
  CACHE STRING "The git tag or hash to checkout for CNPY")

# Where to install CNPY
set(CNPY_CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}"
  CACHE PATH "The installation location of CNPY.")

# The build type for CNPY
set(CNPY_CMAKE_BUILD_TYPE "${CMAKE_BUILD_TYPE}"
  CACHE STRING "The build type for CNPY.")

if (CNPY_CUSTOM_SOURCE_DIR)
  set(CNPY_SOURCE_DIR "${CNPY_CUSTOM_SOURCE_DIR}")
  set(CNPY_URL "")
  set(CNPY_TAG "")
  set(_GIT_REPOSITORY_TAG)
  set(_GIT_TAG_TAG)
  message(STATUS "Using CNPY source in: ${CNPY_SOURCE_DIR}")
else ()
  set(CNPY_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/src")
  set(_GIT_REPOSITORY_TAG "GIT_REPOSITORY")
  set(_GIT_TAG_TAG "GIT_TAG")
endif ()

# Now add the external project
include(ExternalProject)
ExternalProject_Add(CNPY
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}
  TMP_DIR ${CMAKE_CURRENT_BINARY_DIR}/tmp
  STAMP_DIR ${CMAKE_CURRENT_BINARY_DIR}/stamp
  ${_GIT_REPOSITORY_TAG} ${CNPY_URL}
  ${_GIT_TAG_TAG} ${CNPY_TAG}
  SOURCE_DIR ${CNPY_SOURCE_DIR}
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/build
  INSTALL_DIR ${CNPY_CMAKE_INSTALL_PREFIX}
  USES_TERMINAL_BUILD 1
  LOG_DOWNLOAD 1
  LOG_UPDATE 1
  LOG_CONFIGURE 1
  LOG_BUILD 1
  LOG_INSTALL 1
  LOG_TEST 1
  CMAKE_ARGS
  -G${CMAKE_GENERATOR}
  -DCMAKE_INSTALL_PREFIX=${CNPY_CMAKE_INSTALL_PREFIX}
  -DCMAKE_BUILD_TYPE=${CNPY_CMAKE_BUILD_TYPE}
  -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
  -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
  -DENABLE_STATIC=${CNPY_ENABLE_STATIC}
  -DCMAKE_MACOSX_RPATH=ON
  -DBUILD_SHARED_LIBS=${CNPY_BUILD_SHARED_LIBS}
  )

set(CNPY_DIR ${CNPY_CMAKE_INSTALL_PREFIX}
  CACHE INTERNAL "The install prefix of CNPY.")
