enable_language(C)

# FIXME: Currently a NOOP because CUDA+Hydrogen is not done yet.
option(HYDROGEN_USE_CUBLAS "Apply the cuBLAS patch to hydrogen. WARNING: NOOP"
  OFF)

# Handle the clone mechanism. First URL
set(HYDROGEN_URL "https://github.com/llnl/elemental.git"
  CACHE STRING "The URL from which to clone Hydrogen")

# ... then the tag.
set(HYDROGEN_TAG "hydrogen"
  CACHE STRING "The git tag or hash to checkout for Hydrogen")

# Where to install Hydrogen
set(HYDROGEN_CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}"
  CACHE PATH "The installation location of Hydrogen.")

# The build type for Hydrogen
set(HYDROGEN_CMAKE_BUILD_TYPE "${CMAKE_BUILD_TYPE}"
  CACHE STRING "The build type for Hydrogen.")

include(ExternalProject)
ExternalProject_Add(hydrogen
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}
  TMP_DIR ${CMAKE_CURRENT_BINARY_DIR}/tmp
  STAMP_DIR ${CMAKE_CURRENT_BINARY_DIR}/stamp
  GIT_REPOSITORY ${HYDROGEN_URL}
  GIT_TAG ${HYDROGEN_TAG}
  SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/src
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/build
  INSTALL_DIR ${HYDROGEN_CMAKE_INSTALL_PREFIX}
  CMAKE_ARGS
  -G${CMAKE_GENERATOR}
  -DCMAKE_INSTALL_PREFIX=${HYDROGEN_CMAKE_INSTALL_PREFIX}
  -DCMAKE_BUILD_TYPE=${HYDROGEN_CMAKE_BUILD_TYPE}
  -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
  -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
  -DHydrogen_GENERAL_LAPACK_FALLBACK=ON
  )

set(HYDROGEN_DIR ${HYDROGEN_CMAKE_INSTALL_PREFIX}
  CACHE INTERNAL "The install prefix of Hydrogen.")
