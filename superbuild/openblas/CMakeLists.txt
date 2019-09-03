enable_language(C)
enable_language(Fortran)

option(OPENBLAS_USE_64_BIT_INDEX
  "Whether OpenBLAS should use 64-bit indices" OFF)

option(OPENBLAS_USE_OPENMP
  "Whether OpenBLAS should make use of OpenMP" ON)

set(OPENBLAS_MAX_MAKE_JOBS "8"
  CACHE STRING "Number of 'make' jobs allowed for OpenBLAS build.")

set(OPENBLAS_CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}"
  CACHE PATH "The installation location of OpenBLAS.")

option(OPENBLAS_CLONE_VIA_SSH
  "Clone OPENBLAS using SSH instead of HTTPS" ${LBANN_SB_CLONE_VIA_SSH})

if (OPENBLAS_CLONE_VIA_SSH)
  set(OPENBLAS_URL git@github.com:xianyi/OpenBLAS.git
    CACHE STRING "The URL from which to clone OPENBLAS")
else ()
  set(OPENBLAS_URL "https://github.com/xianyi/OpenBLAS.git"
    CACHE STRING "The URL from which to clone OpenBLAS")
endif ()

set(OPENBLAS_TAG "db72ad8"
  CACHE STRING "The git tag or hash to checkout for OpenBLAS")

# Handle the options
if (OPENBLAS_USE_64_BIT_INDEX)
  set(_TMP_OPENBLAS_INTERFACE_COMMAND "INTERFACE64=1 SYMBOLSUFFIX=64")
else()
  set(_TMP_OPENBLAS_INTERFACE_COMMAND "INTERFACE64=0")
endif()

if (OPENBLAS_USE_OPENMP)
  set(_TMP_OPENBLAS_THREAD_COMMAND USE_OPENMP=1)
else()
  set(_TMP_OPENBLAS_THREAD_COMMAND USE_THREAD=0)
endif()

# Arch flags
if(APPLE)
  # This is a hack but is a good default for modern Mac's
  set(_TMP_OPENBLAS_ARCH_COMMAND TARGET=SANDYBRIDGE)
endif()

set(OPENBLAS_INTERFACE_COMMAND "${_TMP_OPENBLAS_INTERFACE_COMMAND}"
  CACHE STRING
  "The command to be forwarded to OpenBLAS to control the index interface.")

set(OPENBLAS_THREAD_COMMAND "${_TMP_OPENBLAS_THREAD_COMMAND}"
  CACHE STRING
  "The command to be forwarded to OpenBLAS to control threading.")

set(OPENBLAS_ARCH_COMMAND "${_TMP_OPENBLAS_ARCH_COMMAND}"
  CACHE STRING
  "THe command to be forwarded to OpenBLAS to describe the CPU architecture.")

# If not using the Makefile generator for CMake, using
# CMAKE_MAKE_PROGRAM probably won't work here (in particular, ninja
# cannot process Makefiles). So we go looking for plain ol' "make"
# instead.
find_program(GNU_MAKE_PROGRAM make)

if (OPENBLAS_CUSTOM_SOURCE_DIR)
  set(OPENBLAS_SOURCE_DIR "${OPENBLAS_CUSTOM_SOURCE_DIR}")
  set(OPENBLAS_URL "")
  set(OPENBLAS_TAG "")
  set(_GIT_REPOSITORY_TAG)
  set(_GIT_TAG_TAG)
  message(STATUS "Using OPENBLAS source in: ${OPENBLAS_SOURCE_DIR}")
else ()
  set(OPENBLAS_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/src")
  set(_GIT_REPOSITORY_TAG "GIT_REPOSITORY")
  set(_GIT_TAG_TAG "GIT_TAG")
endif ()

include (ExternalProject)
ExternalProject_Add(OPENBLAS
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}
  ${_GIT_REPOSITORY_TAG} ${OPENBLAS_URL}
  ${_GIT_TAG_TAG} ${OPENBLAS_TAG}
  SOURCE_DIR ${OPENBLAS_SOURCE_DIR}
  TMP_DIR ${CMAKE_CURRENT_BINARY_DIR}/tmp
  STAMP_DIR ${CMAKE_CURRENT_BINARY_DIR}/stamp
  INSTALL_DIR ${OPENBLAS_CMAKE_INSTALL_PREFIX}
  BUILD_IN_SOURCE 1
  CONFIGURE_COMMAND ""
  UPDATE_COMMAND ""
  USES_TERMINAL_BUILD 1
  LOG_DOWNLOAD 1
  LOG_UPDATE 1
  LOG_CONFIGURE 1
  LOG_BUILD 1
  LOG_INSTALL 1
  LOG_TEST 1
  BUILD_COMMAND ${GNU_MAKE_PROGRAM} -j${OPENBLAS_MAX_MAKE_JOBS}
    CC=${CMAKE_C_COMPILER}
    FC=${CMAKE_Fortran_COMPILER}
    ${OPENBLAS_THREAD_COMMAND}
    ${OPENBLAS_ARCH_COMMAND}
    ${OPENBLAS_INTERFACE_COMMAND}
    libs netlib shared
  INSTALL_COMMAND ${GNU_MAKE_PROGRAM} install PREFIX=<INSTALL_DIR>
  )

set(OPENBLAS_DIR ${OPENBLAS_CMAKE_INSTALL_PREFIX}
  CACHE INTERNAL "The install prefix of OpenBLAS.")
