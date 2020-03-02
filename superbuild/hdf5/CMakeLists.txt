enable_language(C)
enable_language(CXX)

# One weird variable
option(BUILD_HDF5_TESTING "Build HDF5 tests." OFF)

# Standard Superbuild boiler-plate
option(HDF5_CLONE_VIA_SSH
  "Clone HDF5 using SSH instead of HTTPS" OFF)
#${LBANN_SB_CLONE_VIA_SSH})

if (HDF5_CLONE_VIA_SSH)
  message(FATAL_ERROR
    "SSH cloning of HDF5 currently disabled. "
    "I don't know the SSH URL at this time.")

  set(HDF5_URL I_DONT_KNOW
    CACHE STRING "The URL from which to clone HDF5")
else ()
  set(HDF5_URL https://bitbucket.hdfgroup.org/scm/hdffv/hdf5.git
    CACHE STRING "The URL from which to clone HDF5")
endif ()

set(HDF5_TAG "hdf5-1_10_5"
  CACHE STRING "The git tag to checkout for HDF5")

set(HDF5_CMAKE_BUILD_TYPE "${CMAKE_BUILD_TYPE}"
  CACHE STRING "The build type for HDF5.")

set(HDF5_CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}"
  CACHE PATH "The installation location of HDF5.")

option(HDF5_BUILD_SHARED_LIBS
  "Whether HDF5 should build dynamically linked libraries."
  ON)

set(HDF5_CMAKE_COMMAND "${CMAKE_COMMAND}"
  CACHE FILEPATH
  "CMake executable to use with HDF5.")

# HDF5-specific configuration options explicitly exposed
option(HDF5_BUILD_EXAMPLES "Build HDF5 examples." OFF)

option(HDF5_USE_16_API_DEFAULT "Use 1.6 API by default" OFF)
option(HDF5_USE_18_API_DEFAULT "Use 1.8 API by default" ON)
option(HDF5_USE_110_API_DEFAULT "Use 1.10 API by default" OFF)
option(HDF5_USE_112_API_DEFAULT "Use 1.12 API by default" OFF)
option(HDF5_BUILD_FORTRAN "Build HDF5 with fortran support" OFF)
set(HDF5_DEFAULT_API_VERSION "v18")

# At present, this is required for LBANN/JAG use.
option(HDF5_ENABLE_Z_LIB_SUPPORT "Build HDF5 with ZLIB support" ON)

if (HDF5_BUILD_FORTRAN)
  enable_language(Fortran)
endif ()

option(HDF5_ENABLE_PARALLEL "Build HDF5 with parallel (MPI) support." ON)

if (HDF5_ENABLE_PARALLEL)
  # The C++ and Parallel paths are mutually exclusive, apparently.
  set(HDF5_BUILD_CPP_LIB OFF CACHE BOOL "" FORCE)
endif ()

# Get the list of HDF5 variables
get_property(HDF5_VARIABLES DIRECTORY PROPERTY VARIABLES)
list(FILTER HDF5_VARIABLES INCLUDE REGEX
  "^HDF5_.*\|^HDF5_.*\|^LBANN_SB_FWD_HDF5_.*\|^LBANN_SB_FWD_HDF5_.*")
list(FILTER HDF5_VARIABLES EXCLUDE REGEX
  "HDF5_URL\|HDF5_TAG\|HDF5_BUILD_SHARED_LIBS\|HDF5_CLONE_VIA_SSH")

create_cmake_arguments(
  OUTPUT_VARIABLE HDF5_CMAKE_ARGS
  PACKAGE_NAME HDF5
  SKIP_VARS_WITH_PREFIXES "LBANN_SB"
  EXTRA_REMOVE_PREFIXES "LBANN_SB_FWD_HDF5" "LBANN_SB_FWD_HDF5"
  VARIABLES ${HDF5_VARIABLES})

if (HDF5_BUILD_FORTRAN)
  list(APPEND HDF5_CMAKE_ARGS
    "-DCMAKE_Fortran_COMPILER=${CMAKE_Fortran_COMPILER}"
    "-DCMAKE_Fortran_FLAGS=${CMAKE_Fortran_FLAGS}")
endif ()

if (HDF5_CUSTOM_SOURCE_DIR)
  set(HDF5_SOURCE_DIR "${HDF5_CUSTOM_SOURCE_DIR}")
  set(HDF5_URL "")
  set(HDF5_TAG "")
  set(_GIT_REPOSITORY_TAG)
  set(_GIT_TAG_TAG)
  message(STATUS "Using HDF5 source in: ${HDF5_SOURCE_DIR}")
else ()
  set(HDF5_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/src")
  set(_GIT_REPOSITORY_TAG "GIT_REPOSITORY")
  set(_GIT_TAG_TAG "GIT_TAG")
endif ()

include(ExternalProject)
ExternalProject_Add(HDF5
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}
  TMP_DIR ${CMAKE_CURRENT_BINARY_DIR}/tmp
  STAMP_DIR ${CMAKE_CURRENT_BINARY_DIR}/stamp
  ${_GIT_REPOSITORY_TAG} ${HDF5_URL}
  ${_GIT_TAG_TAG} ${HDF5_TAG}
  SOURCE_DIR ${HDF5_SOURCE_DIR}
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/build
  INSTALL_DIR ${HDF5_CMAKE_INSTALL_PREFIX}
  USES_TERMINAL_BUILD 1
  LOG_DOWNLOAD 1
  LOG_UPDATE 1
  LOG_CONFIGURE 1
  LOG_BUILD 1
  LOG_INSTALL 1
  LOG_TEST 1
  LIST_SEPARATOR |
  CMAKE_COMMAND "${HDF5_CMAKE_COMMAND}"
  CMAKE_ARGS
  -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
  -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
  -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
  -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
  -DBUILD_SHARED_LIBS=${HDF5_BUILD_SHARED_LIBS}
  -DBUILD_TESTING=${BUILD_HDF5_TESTING}
  -DHDF5_GENERATE_HEADERS=ON
  -DDEFAULT_API_VERSION=${HDF5_DEFAULT_API_VERSION}
  ${HDF5_CMAKE_ARGS}
  )

set(HDF5_DIR ${HDF5_CMAKE_INSTALL_PREFIX}
  CACHE INTERNAL "The install prefix of HDF5.")
