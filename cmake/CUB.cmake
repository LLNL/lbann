include(ExternalProject)

# Download CUB if it is not found
if(NOT CUB_INCLUDE_DIRS)

  # Git repository URL and tag
  if(NOT CUB_URL)
    set(CUB_URL https://github.com/NVlabs/cub.git)
  endif()
  if(NOT CUB_TAG)
    set(CUB_TAG "1.5.2")
  endif()
  message(STATUS "Will pull CUB (tag ${CUB_TAG}) from ${CUB_URL}")

  # Download and build location
  set(CUB_SOURCE_DIR ${PROJECT_BINARY_DIR}/download/cub/source)
  set(CUB_BINARY_DIR ${PROJECT_BINARY_DIR}/download/cub/build)
  set(CUB_INCLUDE_DIRS ${CUB_SOURCE_DIR})

  # Get CUB from Git repository
  # Note: no compilation is required since CUB is a header library
  ExternalProject_Add(project_CUB
    PREFIX            ${CMAKE_INSTALL_PREFIX}
    TMP_DIR           ${CUB_BINARY_DIR}/tmp
    STAMP_DIR         ${CUB_BINARY_DIR}/stamp
    GIT_REPOSITORY    ${CUB_URL}
    GIT_TAG           ${CUB_TAG}
    SOURCE_DIR        ${CUB_SOURCE_DIR}
    CONFIGURE_COMMAND ""
    BINARY_DIR        ${CUB_BINARY_DIR}
    BUILD_COMMAND     ""
    INSTALL_DIR       ${CMAKE_INSTALL_PREFIX}
    INSTALL_COMMAND   ""
  )

  # LBANN has built CUB
  set(LBANN_BUILT_CUB TRUE)

endif()

# Include CUB header files
include_directories(${CUB_INCLUDE_DIRS})

# LBANN has access to CUB
set(LBANN_HAS_CUB TRUE)