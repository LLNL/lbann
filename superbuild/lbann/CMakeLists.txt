enable_language(CXX)

# Contrary to the top-level CMake, here we make warnings-as-errors the
# default. This should *not* be the persistent norm, but may as well
# save everyone some typing in the meantime.
option(LBANN_WARNINGS_AS_ERRORS "Promote LBANN warnings to errors" ON)

set(LBANN_CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}"
  CACHE PATH "The installation destination of LBANN")

set(LBANN_CMAKE_BUILD_TYPE "${CMAKE_BUILD_TYPE}"
  CACHE STRING "The build type for LBANN.")

# Fixes an RPATH issue with LBANN in which all packages get installed
# to the same prefix
#
# FIXME: Not sure how this will work if CMAKE_INSTALL_LIBDIR is set
# manually for LBANN
if (CMAKE_HOST_SYSTEM_NAME MATCHES "Linux")
  set(_SB_PKGS CNPY CUB ELEMENTAL
    HYDROGEN JPEG_TURBO OPENBLAS OPENCV PROTOBUF)
  foreach(_pkg ${_SB_PKGS})
    if (LBANN_SB_BUILD_${_pkg})
      if (_pkg STREQUAL "HYDROGEN")
        list(APPEND LBANN_CMAKE_INSTALL_RPATH
          "${Hydrogen_CMAKE_INSTALL_PREFIX}/lib")
      else ()
        list(APPEND LBANN_CMAKE_INSTALL_RPATH
          "${${_pkg}_CMAKE_INSTALL_PREFIX}/lib")
      endif ()
    endif ()
  endforeach ()
endif ()
if (LBANN_CMAKE_INSTALL_RPATH)
  list(REMOVE_DUPLICATES LBANN_CMAKE_INSTALL_RPATH)
endif ()

# Build the dependency set
if (TARGET Aluminum)
  option(LBANN_WITH_ALUMINUM "Whether to use Aluminum in LBANN" ON)

  if (LBANN_WITH_ALUMINUM)
    set(LBANN_SB_FWD_LBANN_Aluminum_DIR "${ALUMINUM_DIR}/lib/cmake/Aluminum"
      CACHE STRING "The path to Aluminum for LBANN.")

    list(APPEND _LBANN_DEPENDS Aluminum)
  endif (LBANN_WITH_ALUMINUM)
endif (TARGET Aluminum)
if (TARGET CNPY)
  list(APPEND _LBANN_DEPENDS CNPY)
  set(LBANN_SB_FWD_LBANN_CNPY_DIR "${CNPY_DIR}")
endif ()
if (TARGET CUB)
  list(APPEND _LBANN_DEPENDS CUB)
  set(LBANN_SB_FWD_LBANN_CUB_DIR "${CUB_DIR}")
endif ()
if (TARGET ELEMENTAL)
  if (TARGET OPENBLAS)
    list(APPEND _LBANN_DEPENDS OPENBLAS)
    set(LBANN_SB_FWD_LBANN_BLA_VENDOR "OpenBLAS"
      CACHE STRING "The BLAS vendor for LBANN")
  endif ()
  list(APPEND _LBANN_DEPENDS ELEMENTAL)
  set(LBANN_SB_FWD_LBANN_Elemental_DIR "${ELEMENTAL_DIR}")
endif ()
if (TARGET HYDROGEN)
  if (TARGET OPENBLAS)
    list(APPEND _LBANN_DEPENDS OPENBLAS)
    set(LBANN_SB_FWD_LBANN_BLA_VENDOR "OpenBLAS"
      CACHE STRING "The BLAS vendor for LBANN")
  endif ()
  list(APPEND _LBANN_DEPENDS HYDROGEN)
  set(LBANN_SB_FWD_LBANN_Hydrogen_DIR "${HYDROGEN_DIR}")
endif ()
if (TARGET OPENCV)
  list(APPEND _LBANN_DEPENDS OPENCV)
  set(LBANN_SB_FWD_LBANN_OpenCV_DIR "${OPENCV_DIR}/share/OpenCV")
endif ()
if (TARGET PROTOBUF)
  list(APPEND _LBANN_DEPENDS PROTOBUF)
  set(LBANN_SB_FWD_LBANN_PROTOBUF_DIR "${PROTOBUF_DIR}")
  set(LBANN_SB_FWD_LBANN_protobuf_MODULE_COMPATIBLE ON)
  set(LBANN_SB_FWD_LBANN_protobuf_BUILD_SHARED_LIBS ON)
endif ()

# Get the list of LBANN variables
get_property(LBANN_VARIABLES DIRECTORY PROPERTY VARIABLES)
list(FILTER LBANN_VARIABLES INCLUDE REGEX "^LBANN_.*")

create_cmake_arguments(
  OUTPUT_VARIABLE LBANN_CMAKE_ARGS
  PACKAGE_NAME LBANN
  SKIP_VARS_WITH_PREFIXES "LBANN_SB"
  EXTRA_REMOVE_PREFIXES "LBANN_SB_FWD_LBANN"
  VARIABLES ${LBANN_VARIABLES})

option(LBANN_SB_CLONE_CLEAN_LBANN "Will pull a clean LBANN repository." OFF)

if (LBANN_SB_CLONE_CLEAN_LBANN)

  set(LBANN_URL "https://github.com/llnl/lbann"
    CACHE STRING "The URL from which to clone LBANN")

  set(LBANN_TAG "develop"
    CACHE STRING "The git tag or hash to checkout for LBANN")

  set(LBANN_SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/src)

  set(_git_repo_tag GIT_REPOSITORY)
  set(_git_tag_tag GIT_TAG)

else ()

  unset(LBANN_URL CACHE)
  unset(LBANN_TAG CACHE)
  unset(_git_repo_tag)
  unset(_git_tag_tag)

  get_filename_component(LBANN_SOURCE_DIR ${CMAKE_SOURCE_DIR}/.. REALPATH)

endif ()

include(ExternalProject)
ExternalProject_Add(LBANN
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}
  TMP_DIR ${CMAKE_CURRENT_BINARY_DIR}/tmp
  STAMP_DIR ${CMAKE_CURRENT_BINARY_DIR}/stamp
  ${_git_repo_tag} ${LBANN_URL}
  ${_git_tag_tag} ${LBANN_TAG}
  DEPENDS ${_LBANN_DEPENDS}
  SOURCE_DIR ${LBANN_SOURCE_DIR}
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/build
  INSTALL_DIR ${LBANN_CMAKE_INSTALL_PREFIX}
  USES_TERMINAL_BUILD 1
  LIST_SEPARATOR |
  CMAKE_ARGS
  -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
  -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
  ${LBANN_CMAKE_ARGS}
  )

set(LBANN_DIR ${LBANN_CMAKE_INSTALL_PREFIX}
  CACHE INTERNAL "The install prefix of LBANN.")
