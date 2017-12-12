# Use CNPY_URL to specify the location of the git repo. Use CNPY_TAG
# to specify the commit.

enable_language(C)

# Match the CNPY default
option(CNPY_ENABLE_STATIC "Enable the CNPY static linkage." ON)

# Handle the clone mechanism. First URL
set(CNPY_URL "https://github.com/rogersce/cnpy.git"
  CACHE STRING "The URL from which to clone CNPY")

# ... then the tag.
set(CNPY_TAG "b39d58d3640f77c043bfe92591afeafdd82e78db"
  CACHE STRING "The git tag or hash to checkout for CNPY")

# Where to install CNPY
set(CNPY_CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}"
  CACHE PATH "The installation location of CNPY.")

# The build type for CNPY
set(CNPY_CMAKE_BUILD_TYPE "${CMAKE_BUILD_TYPE}"
  CACHE STRING "The build type for CNPY.")

# Now add the external project
include(ExternalProject)

ExternalProject_Add(CNPY
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}
  TMP_DIR ${CMAKE_CURRENT_BINARY_DIR}/tmp
  STAMP_DIR ${CMAKE_CURRENT_BINARY_DIR}/stamp
  GIT_REPOSITORY ${CNPY_URL}
  GIT_TAG ${CNPY_TAG}
  SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/src
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/build
  INSTALL_DIR ${CNPY_CMAKE_INSTALL_PREFIX}
  CMAKE_ARGS
  -G${CMAKE_GENERATOR}
  -DCMAKE_INSTALL_PREFIX=${CNPY_CMAKE_INSTALL_PREFIX}
  -DCMAKE_BUILD_TYPE=${CNPY_CMAKE_BUILD_TYPE}
  -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
  -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
  -DENABLE_STATIC=${CNPY_ENABLE_STATIC}
  )

set(CNPY_DIR ${CNPY_CMAKE_INSTALL_PREFIX}
  CACHE INTERNAL "The install prefix of CNPY.")
