include(ExternalProject)

if(NOT DEFINED ELEMENTAL_LIBRARY_TYPE)
  set(ELEMENTAL_LIBRARY_TYPE SHARED)
endif()

# Check if Elemental is provided
if(DEFINED CMAKE_ELEMENTAL_DIR)

  message(STATUS "Found Elemental: ${CMAKE_ELEMENTAL_DIR}")

else()

  # Git repository URL and tag
  if(NOT DEFINED ELEMENTAL_URL)
    set(ELEMENTAL_URL "https://github.com/elemental/Elemental.git")
  endif()
  if(NOT DEFINED ELEMENTAL_TAG)
     # Commit from 1/8/2016
     set(ELEMENTAL_TAG "6ec56aaead47848095411a07c73443b036438a72")
  endif()
  message(STATUS "Will pull Elemental (tag ${ELEMENTAL_TAG}) from ${ELEMENTAL_URL}")

  # Elemental options
  if(NOT DEFINED ELEMENTAL_BUILD_TYPE)
    set(ELEMENTAL_BUILD_TYPE "Release")
  endif()
  option(ELEMENTAL_HYBRID "Elemental: make use of OpenMP within MPI packing/unpacking" OFF)
  option(ELEMENTAL_C_INTERFACE "Elemental: build C interface?" OFF)
  option(ELEMENTAL_INSTALL_PYTHON_PACKAGE "Elemental: install Python interface?" OFF)
  option(ELEMENTAL_DISABLE_PARMETIS "Elemental: disable ParMETIS?" ON)

  # Determine library type
  if(ELEMENTAL_LIBRARY_TYPE STREQUAL SHARED)
    set(ELEMENTAL_BUILD_SHARED_LIBS ON)
  elseif(${ELEMENTAL_LIBRARY_TYPE} STREQUAL STATIC)
    set(ELEMENTAL_BUILD_SHARED_LIBS OFF)
  else()
    message(WARNING "Elemental: unknown library type (${ELEMENTAL_LIBRARY_TYPE})")
  endif()

  # Download and build location
  set(ELEMENTAL_SOURCE_DIR "${PROJECT_BINARY_DIR}/download/elemental/source")
  set(ELEMENTAL_BINARY_DIR "${PROJECT_BINARY_DIR}/download/elemental/build")

  # Get Elemental from Git repository and build
  ExternalProject_Add(project_Elemental
    PREFIX          ${CMAKE_INSTALL_PREFIX}
    TMP_DIR         "${ELEMENTAL_BINARY_DIR}/tmp"
    STAMP_DIR       "${ELEMENTAL_BINARY_DIR}/stamp"
    GIT_REPOSITORY  ${ELEMENTAL_URL}
    GIT_TAG         ${ELEMENTAL_TAG}
    SOURCE_DIR      ${ELEMENTAL_SOURCE_DIR}
    BINARY_DIR      ${ELEMENTAL_BINARY_DIR}
    BUILD_COMMAND   ${CMAKE_MAKE_PROGRAM} -j${MAKE_NUM_PROCESSES}
    INSTALL_DIR     ${CMAKE_INSTALL_PREFIX}
    INSTALL_COMMAND ${CMAKE_MAKE_PROGRAM} install -j${MAKE_NUM_PROCESSES}
    CMAKE_ARGS
      -D CMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
      -D CMAKE_BUILD_TYPE=${ELEMENTAL_BUILD_TYPE}
      -D CMAKE_C_COMPILER=${CMAKE_C_COMPILER}
      -D CMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
      -D CMAKE_Fortran_COMPILER=${CMAKE_Fortran_COMPILER}
      -D MPI_C_COMPILER=${MPI_C_COMPILER}
      -D MPI_CXX_COMPILER=${MPI_CXX_COMPILER}
      -D MPI_Fortran_COMPILER=${MPI_Fortran_COMPILER}
      -D MATH_LIBS=${MATH_LIBS}
      -D BUILD_SHARED_LIBS=${ELEMENTAL_BUILD_SHARED_LIBS}
      -D EL_HYBRID=${ELEMENTAL_HYBRID}
      -D EL_C_INTERFACE=${ELEMENTAL_C_INTERFACE}
      -D INSTALL_PYTHON_PACKAGE=${ELEMENTAL_INSTALL_PYTHON_PACKAGE}
      -D EL_DISABLE_PARMETIS=${ELEMENTAL_DISABLE_PARMETIS}
  )

  # Get install directory
  set(CMAKE_ELEMENTAL_DIR "${CMAKE_INSTALL_PREFIX}")

  # LBANN has built Elemental
  set(LBANN_BUILT_ELEMENTAL TRUE)

endif()

# Get header files
set(ELEMENTAL_INCLUDE_DIRS "${CMAKE_ELEMENTAL_DIR}/include")
include_directories(${ELEMENTAL_INCLUDE_DIRS})

# Get library
if(ELEMENTAL_SHARED_LIBS STREQUAL STATIC)
  add_library(libElemental STATIC IMPORTED)
  set(ELEMENTAL_LIBRARIES "${CMAKE_ELEMENTAL_DIR}/lib/libEl.a")
else()
  add_library(libElemental SHARED IMPORTED)
  set(ELEMENTAL_LIBRARIES "${CMAKE_ELEMENTAL_DIR}/lib/libEl.so")
endif()
set_property(TARGET libElemental PROPERTY IMPORTED_LOCATION ${ELEMENTAL_LIBRARIES})
link_directories("${CMAKE_ELEMENTAL_DIR}/lib")
#  set(CMAKE_CXX_FLAGS "${MATH_LIBS} ${CMAKE_CXX_FLAGS}")

# Add preprocessor flag for Elemental
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__LIB_ELEMENTAL")

# LBANN has access to Elemental
set(LBANN_HAS_ELEMENTAL TRUE)
