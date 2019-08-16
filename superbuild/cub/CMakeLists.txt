option(CUB_CLONE_VIA_SSH
  "Clone CUB using SSH instead of HTTPS" ${LBANN_SB_CLONE_VIA_SSH})

if (CUB_CLONE_VIA_SSH)
  set(CUB_URL git@github.com:NVlabs/cub.git
    CACHE STRING "The URL from which to clone CUB")
else ()
  set(CUB_URL https://github.com/NVlabs/cub.git
    CACHE STRING "The URL from which to clone CUB.")
endif ()

set(CUB_TAG "1.8.0" CACHE STRING "The git tag or hash to checkout for CUB")

# Where to install CUB
set(CUB_CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}"
  CACHE PATH "The installation location of CUB.")

# Handle the install of CUB
include(ExternalProject)

if (CUB_CUSTOM_SOURCE_DIR)
  set(CUB_SOURCE_DIR "${CUB_CUSTOM_SOURCE_DIR}")
  set(CUB_URL "")
  set(CUB_TAG "")
  set(_GIT_REPOSITORY_TAG)
  set(_GIT_TAG_TAG)
  message(STATUS "Using CUB source in: ${CUB_SOURCE_DIR}")
else ()
  set(CUB_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/src")
  set(_GIT_REPOSITORY_TAG "GIT_REPOSITORY")
  set(_GIT_TAG_TAG "GIT_TAG")
endif ()

ExternalProject_Add(CUB
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}
  TMP_DIR ${CMAKE_CURRENT_BINARY_DIR}/tmp
  STAMP_DIR ${CMAKE_CURRENT_BINARY_DIR}/stamp
  ${_GIT_REPOSITORY_TAG} ${CUB_URL}
  ${_GIT_TAG_TAG} ${CUB_TAG}
  SOURCE_DIR ${CUB_SOURCE_DIR}
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/build
  PATCH_COMMAND
  ${CMAKE_COMMAND} -E copy_if_different
  ${CMAKE_CURRENT_SOURCE_DIR}/CUBCMakeLists.txt
  ${CUB_SOURCE_DIR}/CMakeLists.txt &&
  patch -p1 < ${LBANN_SRC_DIR}/external/cub/cub_enable_alloc_free_logging.patch
  INSTALL_DIR ${CUB_CMAKE_INSTALL_PREFIX}
  USES_TERMINAL_BUILD 1
  LOG_DOWNLOAD 1
  LOG_UPDATE 1
  LOG_CONFIGURE 1
  LOG_BUILD 1
  LOG_INSTALL 1
  LOG_TEST 1
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CUB_CMAKE_INSTALL_PREFIX}
  )

set(CUB_DIR ${CUB_CMAKE_INSTALL_PREFIX}
  CACHE INTERNAL "The install prefix of CUB.")
