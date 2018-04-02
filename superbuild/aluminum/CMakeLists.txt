message(FATAL_ERROR "Aluminum not yet available in superbuild.")
set(ALUMINUM_URL https://github.com/ndryden/Aluminum
  CACHE STRING "The URL from which to clone Aluminum")

set(ALUMINUM_TAG "master" 
  CACHE STRING "The git tag to checkout for Aluminum")

# Where to install LIBALUMINUM
set (ALUMINUM_CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}"
  CACHE PATH "The installation location of Aluminum.")

set(ALUMINUM_MAKE_JOBS "8"
  CACHE INTERNAL "The number of make jobs to run. Max by default.")


find_program(GNU_MAKE_PROGRAM make)

# Handle the install of 
include(ExternalProject)
ExternalProject_Add(ALUMINUM
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}
  TMP_DIR ${CMAKE_CURRENT_BINARY_DIR}/tmp
  STAMP_DIR ${CMAKE_CURRENT_BINARY_DIR}/stamp
  GIT_REPOSITORY ${ALUMINUM_URL}
  GIT_TAG ${ALUMINUM_TAG}
  SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/src
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/build
  INSTALL_DIR ${ALUMINUM_CMAKE_INSTALL_PREFIX}
  BUILD_COMMAND  ${GNU_MAKE_PROGRAM} -j${ALUMINUM_MAKE_JOBS} 
  
  )

  set(ALUMINUM_DIR ${ALUMINUM_CMAKE_INSTALL_PREFIX}
    CACHE INTERNAL "The install prefix of Aluminum.")
