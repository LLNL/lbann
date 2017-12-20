enable_language(C)
enable_language(CXX)

# Where to install Hydrogen
set(Hydrogen_CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}"
  CACHE PATH "The installation location of Hydrogen.")

# The build type for Hydrogen
set(Hydrogen_CMAKE_BUILD_TYPE "${CMAKE_BUILD_TYPE}"
  CACHE STRING "The build type for Hydrogen.")

if (TARGET OPENBLAS)
  set(_hydrogen_depends_tag DEPENDS)
  list(APPEND _HYDROGEN_DEPENDS OPENBLAS)
  set(Hydrogen_USE_OpenBLAS ON
    CACHE BOOL "Whether to use OpenBLAS in Hydrogen")
  set(LBANN_SB_FWD_HYDROGEN_OpenBLAS_DIR "${OPENBLAS_DIR}"
    CACHE STRING "The path to OpenBLAS for Hydrogen.")
endif ()

# Get the list of HYDROGEN variables
get_property(HYDROGEN_VARIABLES DIRECTORY PROPERTY VARIABLES)
list(FILTER HYDROGEN_VARIABLES INCLUDE REGEX "^HYDROGEN_.*\|^Hydrogen_.*\|^LBANN_SB_FWD_HYDROGEN_.*")
list(FILTER HYDROGEN_VARIABLES EXCLUDE REGEX "Hydrogen_URL\|Hydrogen_TAG\|Hydrogen_CMAKE_GENERATOR")

create_cmake_arguments(
  OUTPUT_VARIABLE HYDROGEN_CMAKE_ARGS
  PACKAGE_NAME Hydrogen
  SKIP_VARS_WITH_PREFIXES "LBANN_SB"
  EXTRA_REMOVE_PREFIXES "LBANN_SB_FWD_HYDROGEN" "LBANN_SB_FWD_Hydrogen"
  VARIABLES ${HYDROGEN_VARIABLES})

# Set the generator
set(Hydrogen_CMAKE_GENERATOR "${CMAKE_GENERATOR}"
  CACHE STRING "The generator used by CMake for Hydrogen.")

# Handle the clone mechanism. First URL
set(Hydrogen_URL "https://github.com/llnl/elemental.git"
  CACHE STRING "The URL from which to clone Hydrogen")

# ... then the tag.
set(Hydrogen_TAG "hydrogen"
  CACHE STRING "The git tag or hash to checkout for Hydrogen")


include(ExternalProject)
ExternalProject_Add(HYDROGEN
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}
  TMP_DIR ${CMAKE_CURRENT_BINARY_DIR}/tmp
  STAMP_DIR ${CMAKE_CURRENT_BINARY_DIR}/stamp
  GIT_REPOSITORY ${Hydrogen_URL}
  GIT_TAG ${Hydrogen_TAG}
  ${_hydrogen_depends_tag} ${_HYDROGEN_DEPENDS}
  SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/src
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/build
  INSTALL_DIR ${Hydrogen_CMAKE_INSTALL_PREFIX}
  USES_TERMINAL_BUILD 1
  CMAKE_ARGS
  -G${Hydrogen_CMAKE_GENERATOR}
  -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
  -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
  ${HYDROGEN_CMAKE_ARGS}
  )

set(HYDROGEN_DIR ${Hydrogen_CMAKE_INSTALL_PREFIX}
  CACHE INTERNAL "The install prefix of Hydrogen.")
