# This library is header-only. Few options. Still need to pass C and
# CXX compilers though. :/
enable_language(C)

option(CEREAL_CLONE_VIA_SSH
  "Clone CEREAL using SSH instead of HTTPS" ${LBANN_SB_CLONE_VIA_SSH})

if (CEREAL_CLONE_VIA_SSH)
  set(CEREAL_URL git@github.com:uscilab/cereal.git
    CACHE STRING "The URL from which to clone CEREAL")
else ()
  set(CEREAL_URL https://github.com/uscilab/cereal.git
    CACHE STRING "The URL from which to clone CEREAL.")
endif ()

set(CEREAL_TAG "master" CACHE STRING "The git tag or hash to checkout for CEREAL")

# Where to install CEREAL
set(CEREAL_CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}"
  CACHE PATH "The installation location of CEREAL.")

# Handle the install of CEREAL
include(ExternalProject)

ExternalProject_Add(CEREAL
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}
  TMP_DIR ${CMAKE_CURRENT_BINARY_DIR}/tmp
  STAMP_DIR ${CMAKE_CURRENT_BINARY_DIR}/stamp
  GIT_REPOSITORY ${CEREAL_URL}
  GIT_TAG ${CEREAL_TAG}
  SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/src
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/build
  INSTALL_DIR ${CEREAL_CMAKE_INSTALL_PREFIX}
  USES_TERMINAL_BUILD 1
  LOG_DOWNLOAD 1
  LOG_UPDATE 1
  LOG_CONFIGURE 1
  LOG_BUILD 1
  LOG_INSTALL 1
  LOG_TEST 1
  CMAKE_ARGS
  -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
  -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
  -DCMAKE_INSTALL_PREFIX=${CEREAL_CMAKE_INSTALL_PREFIX}
  -DJUST_INSTALL_CEREAL=ON
  -DWITH_WERROR=OFF
  )

set(CEREAL_DIR ${CEREAL_CMAKE_INSTALL_PREFIX}
  CACHE INTERNAL "The install prefix of CEREAL.")
