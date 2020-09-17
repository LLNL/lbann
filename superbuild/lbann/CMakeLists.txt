enable_language(CXX)

option(LBANN_BUILD_SHARED_LIBS
  "Build liblbann as a shared library."
  ${BUILD_SHARED_LIBS})

# Contrary to the top-level CMake, here we make warnings-as-errors the
# default. This should *not* be the persistent norm, but may as well
# save everyone some typing in the meantime.
option(LBANN_WARNINGS_AS_ERRORS
  "Promote LBANN warnings to errors" OFF)

set(LBANN_CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}"
  CACHE PATH "The installation destination of LBANN")

set(LBANN_CMAKE_BUILD_TYPE "${CMAKE_BUILD_TYPE}"
  CACHE STRING "The build type for LBANN.")

if (Hydrogen_ENABLE_CUDA OR LBANN_WITH_CUDA OR CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  set(LBANN_CMAKE_CUDA_COMPILER ${CMAKE_CUDA_COMPILER})
  set(LBANN_CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS})
  if (NOT CMAKE_CUDA_HOST_COMPILER)
    set(LBANN_CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
  else ()
    set(LBANN_CMAKE_CUDA_HOST_COMPILER ${CMAKE_CUDA_HOST_COMPILER})
  endif ()
endif ()

if (LBANN_WITH_CONDUIT)
  enable_language(C)
  set(LBANN_CMAKE_C_COMPILER ${CMAKE_C_COMPILER})
  set(LBANN_CMAKE_C_FLAGS ${CMAKE_C_FLAGS})
endif ()

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
          "${HYDROGEN_CMAKE_INSTALL_PREFIX}/lib")
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
if (TARGET ALUMINUM)
  option(LBANN_WITH_ALUMINUM "Whether to use Aluminum in LBANN" ON)

  if (LBANN_WITH_ALUMINUM)
    set(LBANN_SB_FWD_LBANN_Aluminum_DIR "${ALUMINUM_DIR}/lib/cmake/Aluminum"
      CACHE STRING "The path to Aluminum for LBANN.")

    list(APPEND _LBANN_DEPENDS ALUMINUM)
  endif (LBANN_WITH_ALUMINUM)
endif (TARGET ALUMINUM)

if (TARGET CATCH2)
  option(LBANN_WITH_UNIT_TESTING "Build catch2 unit tests in LBANN" ON)
  if (LBANN_WITH_UNIT_TESTING)
    set(LBANN_SB_FWD_LBANN_CATCH2_DIR ${CATCH2_DIR})
    list(APPEND _LBANN_DEPENDS CATCH2)
  endif ()
endif (TARGET CATCH2)

if (TARGET CEREAL)
  list(APPEND _LBANN_DEPENDS CEREAL)
  set(LBANN_SB_FWD_LBANN_CEREAL_DIR "${CEREAL_DIR}")
endif ()
if (TARGET CNPY)
  list(APPEND _LBANN_DEPENDS CNPY)
  set(LBANN_SB_FWD_LBANN_CNPY_DIR "${CNPY_DIR}")
endif ()
if (TARGET CONDUIT)
  option(LBANN_WITH_CONDUIT "Whether to use CONDUIT in LBANN" ON)

  list(APPEND _LBANN_DEPENDS CONDUIT)
  set(LBANN_SB_FWD_LBANN_CONDUIT_DIR "${CONDUIT_DIR}")
endif ()
if (TARGET CUB)
  list(APPEND _LBANN_DEPENDS CUB)
  set(LBANN_SB_FWD_LBANN_CUB_DIR "${CUB_DIR}")
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
  set(LBANN_SB_FWD_LBANN_protobuf_BUILD_SHARED_LIBS
    ${PROTOBUF_BUILD_SHARED_LIBS})

  if (TARGET HOST-PROTOBUF)
    list(APPEND _LBANN_DEPENDS HOST-PROTOBUF)
    set(LBANN_USE_PROTOBUF_MODULE ON)
    set(LBANN_SB_FWD_LBANN_Protobuf_PROTOC_EXECUTABLE
      "${HOST_PROTOBUF_protoc_EXE}")
    # This will exist by the time LBANN configures.
  endif ()
endif ()

# DiHydrogen and Distconv
if (TARGET DIHYDROGEN)
  option(LBANN_WITH_DIHYDROGEN "Whether to use DiHydrogen in LBANN" ON)
  if (LBANN_WITH_DIHYDROGEN)
    list(APPEND _LBANN_DEPENDS DIHYDROGEN)
    set(LBANN_SB_FWD_LBANN_DIHYDROGEN_DIR "${DIHYDROGEN_DIR}"
      CACHE STRING "The path to DiHydrogen for LBANN.")
    set(LBANN_SB_FWD_LBANN_WITH_DIHYDROGEN ON)
  endif ()
  option(LBANN_WITH_DISTCONV "Whether to use Distconv in LBANN" OFF)
  if (LBANN_WITH_DISTCONV)
    set(LBANN_SB_FWD_LBANN_WITH_DISTCONV ON)
  endif ()
endif ()

# Get the list of LBANN variables
get_property(LBANN_VARIABLES DIRECTORY PROPERTY VARIABLES)
list(FILTER LBANN_VARIABLES INCLUDE REGEX "^LBANN_.*")
list(FILTER LBANN_VARIABLES EXCLUDE REGEX
  "LBANN_URL\|LBANN_TAG\|LBANN_CMAKE_GENERATOR\|LBANN_BUILD_SHARED_LIBS\|LBANN_CLONE_VIA_SSH")

create_cmake_arguments(
  OUTPUT_VARIABLE LBANN_CMAKE_ARGS
  PACKAGE_NAME LBANN
  SKIP_VARS_WITH_PREFIXES "LBANN_SB"
  EXTRA_REMOVE_PREFIXES "LBANN_SB_FWD_LBANN"
  VARIABLES ${LBANN_VARIABLES})

option(LBANN_SB_CLONE_CLEAN_LBANN "Will pull a clean LBANN repository." OFF)

if (LBANN_SB_CLONE_CLEAN_LBANN)
  option(LBANN_CLONE_VIA_SSH
    "Clone LBANN using SSH instead of HTTPS" ${LBANN_SB_CLONE_VIA_SSH})

  if (LBANN_CLONE_VIA_SSH)
    set(LBANN_URL git@github.com:LLNL/lbann.git
      CACHE STRING "The URL from which to clone LBANN")
  else ()
    set(LBANN_URL "https://github.com/llnl/lbann"
      CACHE STRING "The URL from which to clone LBANN")
  endif ()

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
  LOG_DOWNLOAD 1
  LOG_UPDATE 1
  LOG_CONFIGURE 1
  LOG_BUILD 1
  LOG_INSTALL 1
  LOG_TEST 1
  LIST_SEPARATOR |
  CMAKE_ARGS
  -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
  -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
  -DBUILD_SHARED_LIBS=${LBANN_BUILD_SHARED_LIBS}
  ${LBANN_CMAKE_ARGS}
  )

# Ensure the JAG utils are built
ExternalProject_Add_Step(LBANN build-jag-utils
  COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --config $<CONFIG> --target jag-utils
  COMMENT "Performing building of JAG utils for 'LBANN'"
  DEPENDEES build
  DEPENDERS install
  LOG 1
  USES_TERMINAL 1)
# Ensure the ATOM utils are built
ExternalProject_Add_Step(LBANN build-atom-utils
  COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --config $<CONFIG> --target atom-utils
  COMMENT "Performing building of ATOM tools for 'LBANN'"
  DEPENDEES build
  DEPENDERS install
  LOG 1
  USES_TERMINAL 1)

set(LBANN_DIR ${LBANN_CMAKE_INSTALL_PREFIX}
  CACHE INTERNAL "The install prefix of LBANN.")
