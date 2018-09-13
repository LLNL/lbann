set(Aluminum_URL https://github.com/ndryden/Aluminum
  CACHE STRING "The URL from which to clone Aluminum")

set(Aluminum_TAG "master"
  CACHE STRING "The git tag to checkout for Aluminum")

set(Aluminum_CMAKE_BUILD_TYPE "${CMAKE_BUILD_TYPE}"
  CACHE STRING "The build type for Aluminum.")

# Where to install Aluminum
set (Aluminum_CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}"
  CACHE PATH "The installation location of Aluminum.")

# Configuration options explicitly exposed
option(ALUMINUM_ENABLE_CUDA "Enable CUDA support." OFF)
option(ALUMINUM_ENABLE_MPI_CUDA "Enable MPI-CUDA support." OFF)
option(ALUMINUM_ENABLE_NCCL "Enable NCCL support." OFF)

# Get the list of ALUMINUM variables
get_property(ALUMINUM_VARIABLES DIRECTORY PROPERTY VARIABLES)
list(FILTER ALUMINUM_VARIABLES INCLUDE REGEX
  "^ALUMINUM_.*\|^Aluminum_.*\|^LBANN_SB_FWD_ALUMINUM_.*\|^LBANN_SB_FWD_Aluminum_.*\|CMAKE_\(CXX\|CUDA\)_COMPILER\|CMAKE_BUILD_TYPE")
list(FILTER ALUMINUM_VARIABLES EXCLUDE REGEX "Aluminum_URL\|Aluminum_TAG")

create_cmake_arguments(
  OUTPUT_VARIABLE ALUMINUM_CMAKE_ARGS
  PACKAGE_NAME Aluminum
  SKIP_VARS_WITH_PREFIXES "LBANN_SB"
  EXTRA_REMOVE_PREFIXES "LBANN_SB_FWD_ALUMINUM" "LBANN_SB_FWD_Aluminum"
  VARIABLES ${ALUMINUM_VARIABLES})

include(ExternalProject)
ExternalProject_Add(Aluminum
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}
  TMP_DIR ${CMAKE_CURRENT_BINARY_DIR}/tmp
  STAMP_DIR ${CMAKE_CURRENT_BINARY_DIR}/stamp
  GIT_REPOSITORY ${Aluminum_URL}
  GIT_TAG ${Aluminum_TAG}
  SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/src
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/build
  INSTALL_DIR ${Aluminum_CMAKE_INSTALL_PREFIX}
  USES_TERMINAL_BUILD 1
  LIST_SEPARATOR |
  CMAKE_ARGS
  ${ALUMINUM_CMAKE_ARGS}
  )

set(ALUMINUM_DIR ${Aluminum_CMAKE_INSTALL_PREFIX}
  CACHE INTERNAL "The install prefix of Aluminum.")
