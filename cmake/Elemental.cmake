include(ExternalProject)

# Download and build Elemental if it is not found
if(NOT ELEMENTAL_FOUND)

  # Git repository URL and tag
  if(NOT DEFINED ELEMENTAL_URL)
    set(ELEMENTAL_URL "https://github.com/elemental/Elemental.git")
  endif()
  if(NOT DEFINED ELEMENTAL_TAG)
     # Commit from 8/29/2016
     # set(ELEMENTAL_TAG "85572b9c79fab694457bf133e4b01289d4cdcd21")
     # Commit from 6/8/2016
     set(ELEMENTAL_TAG "4748937")
  endif()
  message(STATUS "Will pull Elemental (tag ${ELEMENTAL_TAG}) from ${ELEMENTAL_URL}")

  # Elemental options
  if(NOT DEFINED ELEMENTAL_BUILD_TYPE)
    set(ELEMENTAL_BUILD_TYPE "Release")
  endif()
  option(ELEMENTAL_BUILD_SHARED_LIBS "Elemental: build with shared libraries?" ON)
  option(ELEMENTAL_HYBRID "Elemental: make use of OpenMP within MPI packing/unpacking" OFF)
  option(ELEMENTAL_C_INTERFACE "Elemental: build C interface?" OFF)
  option(ELEMENTAL_INSTALL_PYTHON_PACKAGE "Elemental: install Python interface?" OFF)
  option(ELEMENTAL_DISABLE_PARMETIS "Elemental: disable ParMETIS?" OFF)

  # TODO: make nice
  set(MATH_LIBS "-L/usr/gapps/brain/installs/BLAS/surface/lib -lopenblas")

  # Download and build location
  set(ELEMENTAL_SOURCE_DIR "${PROJECT_BINARY_DIR}/download/elemental/source")
  set(ELEMENTAL_BINARY_DIR "${PROJECT_BINARY_DIR}/download/elemental/build")

  # Get Elemental from Git repository and build
  ExternalProject_Add(project_Elemental
    PREFIX         ${CMAKE_INSTALL_PREFIX}
    TMP_DIR        "${ELEMENTAL_BINARY_DIR}/tmp"
    STAMP_DIR      "${ELEMENTAL_BINARY_DIR}/stamp"
    GIT_REPOSITORY ${ELEMENTAL_URL}
    GIT_TAG        ${ELEMENTAL_TAG}
    SOURCE_DIR     ${ELEMENTAL_SOURCE_DIR}
    BINARY_DIR     ${ELEMENTAL_BINARY_DIR}
    INSTALL_DIR    ${CMAKE_INSTALL_PREFIX}
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
      -D CMAKE_CXX_FLAGS="-I/usr/gapps/brain/installs/BLAS/surface/include" # TODO: remove
  )

  # Get header files
  ExternalProject_Get_Property(project_Elemental install_dir)
  set(ELEMENTAL_INCLUDE_DIRS "${install_dir}/include")

  # Get library
  add_library(libelemental SHARED IMPORTED)
  if(ELEMENTAL_BUILD_SHARED_LIBS)
    set(ELEMENTAL_LIBRARIES "${install_dir}/lib/libelemental.so")
  else()
    set(ELEMENTAL_LIBRARIES "${install_dir}/lib/libelemental.a")
  endif()
  set_property(TARGET libelemental PROPERTY IMPORTED_LOCATION ${ELEMENTAL_LIBRARIES})
  link_directories("${install_dir}/lib")

  # LBANN has built Elemental
  set(LBANN_BUILT_ELEMENTAL TRUE)

endif()

# Include Elemental header files
include_directories(${ELEMENTAL_INCLUDE_DIRS})

# Add preprocessor flag for Elemental
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__LIB_ELEMENTAL")

# LBANN has access to Elemental
set(LBANN_HAS_ELEMENTAL TRUE)
