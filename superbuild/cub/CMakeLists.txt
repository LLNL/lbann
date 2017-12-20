set(CUB_URL https://github.com/NVlabs/cub.git
  CACHE STRING "The URL from which to clone CUB.")

set(CUB_TAG "1.5.2" CACHE STRING "The git tag or hash to checkout for CUB")

# Where to install CUB
set(CUB_CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}"
  CACHE PATH "The installation location of CUB.")

# Handle the install of CUB
include(ExternalProject)

ExternalProject_Add(CUB
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}
  TMP_DIR ${CMAKE_CURRENT_BINARY_DIR}/tmp
  STAMP_DIR ${CMAKE_CURRENT_BINARY_DIR}/stamp
  GIT_REPOSITORY ${CUB_URL}
  GIT_TAG ${CUB_TAG}
  SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/src
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/build
  PATCH_COMMAND
  ${CMAKE_COMMAND} -E copy_if_different
  ${CMAKE_CURRENT_SOURCE_DIR}/CUBCMakeLists.txt
  ${CMAKE_CURRENT_BINARY_DIR}/src/CMakeLists.txt
  INSTALL_DIR ${CUB_CMAKE_INSTALL_PREFIX}
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CUB_CMAKE_INSTALL_PREFIX}
  )

set(CUB_DIR ${CUB_CMAKE_INSTALL_PREFIX}
  CACHE INTERNAL "The install prefix of CUB.")
