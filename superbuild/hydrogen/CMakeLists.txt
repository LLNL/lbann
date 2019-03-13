enable_language(CXX)

option(Hydrogen_ENABLE_CUDA "Enable CUDA support in Hydrogen" OFF)

option(Hydrogen_ENABLE_OPENMP "Hydrogen use OpenMP threading." ON)
option(Hydrogen_ENABLE_OMP_TASKLOOP "Hydrogen use OpenMP taskloops for parallel regions." OFF)

option(Hydrogen_USE_64BIT_INTS
  "Hydrogen use 64-bit internal indices" ON)

option(Hydrogen_USE_64BIT_BLAS_INTS
  "Hydrogen use 64-bit integers in BLAS calls" OFF)

option(Hydrogen_ENABLE_TESTING "Enable the hydrogen tests." OFF)

# Where to install Hydrogen
set(HYDROGEN_CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}"
  CACHE PATH "The installation location of Hydrogen.")

# The build type for Hydrogen
set(HYDROGEN_CMAKE_BUILD_TYPE "${CMAKE_BUILD_TYPE}"
  CACHE STRING "The build type for Hydrogen.")

option(Hydrogen_BUILD_SHARED_LIBS
  "Build dynamically linked libraries for Hydrogen" ${BUILD_SHARED_LIBS})

if (TARGET OPENBLAS)
  set(_hydrogen_depends_tag DEPENDS)
  list(APPEND _HYDROGEN_DEPENDS OPENBLAS)
  set(Hydrogen_USE_OpenBLAS ON
    CACHE BOOL "Whether to use OpenBLAS in Hydrogen")
  set(LBANN_SB_FWD_HYDROGEN_OpenBLAS_DIR "${OPENBLAS_DIR}/lib/cmake/openblas"
    CACHE STRING "The path to OpenBLAS for Hydrogen.")

  if (OPENBLAS_USE_64_BIT_INDEX)
    set(Hydrogen_USE_64BIT_BLAS_INTS ON)
  endif ()
endif ()

if (TARGET ALUMINUM)
  option(Hydrogen_ENABLE_ALUMINUM "Whether to use Aluminum in Hydrogen" ON)

  if (Hydrogen_ENABLE_ALUMINUM)
    message(STATUS "Building Hydrogen with Aluminum support")
    set(LBANN_SB_FWD_HYDROGEN_Aluminum_DIR
      "${ALUMINUM_DIR}/lib/cmake/aluminum"
      CACHE STRING "The path to Aluminum for Hydrogen.")

    set(_hydrogen_depends_tag DEPENDS)
    list(APPEND _HYDROGEN_DEPENDS ALUMINUM)

    if (ALUMINUM_ENABLE_CUDA OR ALUMINUM_ENABLE_MPI_CUDA OR ALUMINUM_ENABLE_NCCL)
      set(Hydrogen_ENABLE_CUDA ON)
    endif ()
  endif (Hydrogen_ENABLE_ALUMINUM)
endif (TARGET ALUMINUM)

if (Hydrogen_ENABLE_CUDA)
  enable_language(CUDA)
endif ()

# Get the list of HYDROGEN variables
get_property(HYDROGEN_VARIABLES DIRECTORY PROPERTY VARIABLES)
list(FILTER HYDROGEN_VARIABLES INCLUDE REGEX
  "^HYDROGEN_.*\|^Hydrogen_.*\|^LBANN_SB_FWD_HYDROGEN_.*\|^LBANN_SB_FWD_Hydrogen_.*")
list(FILTER HYDROGEN_VARIABLES EXCLUDE REGEX
  "Hydrogen_URL\|Hydrogen_TAG\|Hydrogen_CMAKE_GENERATOR\|Hydrogen_BUILD_SHARED_LIBS")

create_cmake_arguments(
  OUTPUT_VARIABLE HYDROGEN_CMAKE_ARGS
  PACKAGE_NAME HYDROGEN
  SKIP_VARS_WITH_PREFIXES "LBANN_SB"
  EXTRA_REMOVE_PREFIXES "LBANN_SB_FWD_HYDROGEN" "LBANN_SB_FWD_Hydrogen"
  VARIABLES ${HYDROGEN_VARIABLES})

# Handle the clone mechanism. First URL
option(HYDROGEN_CLONE_VIA_SSH
  "Clone Hydrogen using SSH instead of HTTPS" ${LBANN_SB_CLONE_VIA_SSH})

if (HYDROGEN_CLONE_VIA_SSH)
  set(HYDROGEN_URL git@github.com:LLNL/Elemental.git
    CACHE STRING "The URL from which to clone Hydrogen")
else ()
  set(HYDROGEN_URL "https://github.com/LLNL/Elemental.git"
    CACHE STRING "The URL from which to clone Hydrogen")
endif ()

# ... then the tag.
set(HYDROGEN_TAG "hydrogen"
  CACHE STRING "The git tag or hash to checkout for Hydrogen")

include(ExternalProject)
ExternalProject_Add(HYDROGEN
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}
  TMP_DIR ${CMAKE_CURRENT_BINARY_DIR}/tmp
  STAMP_DIR ${CMAKE_CURRENT_BINARY_DIR}/stamp
  GIT_REPOSITORY ${HYDROGEN_URL}
  GIT_TAG ${HYDROGEN_TAG}
  ${_hydrogen_depends_tag} ${_HYDROGEN_DEPENDS}
  SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/src
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/build
  INSTALL_DIR ${HYDROGEN_CMAKE_INSTALL_PREFIX}
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
  -DCMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}
  -DCMAKE_CUDA_HOST_COMPILER=${CMAKE_CUDA_HOST_COMPILER}
  -DBUILD_SHARED_LIBS=${Hydrogen_BUILD_SHARED_LIBS}
  ${HYDROGEN_CMAKE_ARGS}
  )

set(HYDROGEN_DIR ${HYDROGEN_CMAKE_INSTALL_PREFIX}
  CACHE INTERNAL "The install prefix of Hydrogen.")
