include(ExternalProject)

# Download and build Elemental if it is not found
if(NOT ELEMENTAL_FOUND)

  # Git repository URL and tag
  if(NOT DEFINED ELEMENTAL_URL)
    set(ELEMENTAL_URL "https://github.com/elemental/Elemental.git")
  endif()
  if(NOT DEFINED ELEMENTAL_TAG)
     # Commit from 8/29/2016
     set(ELEMENTAL_TAG "85572b9c79fab694457bf133e4b01289d4cdcd21")
     # Commit from 6/24/2016
     # set(ELEMENTAL_TAG "a95d62152fb9eb25dfdbc67e24097108cd5f9d7d")
  endif()
  message(STATUS "Will pull Elemental (tag ${ELEMENTAL_TAG}) from ${ELEMENTAL_URL}")

  # Elemental options
  if(NOT DEFINED ELEMENTAL_BUILD_TYPE)
    set(ELEMENTAL_BUILD_TYPE "Release")
  endif()
  if(NOT DEFINED ELEMENTAL_INSTALL_PYTHON_PACKAGE)
    set(ELEMENTAL_INSTALL_PYTHON_PACKAGE OFF)
  endif()
  if(NOT DEFINED ELEMENTAL_HYBRID)
    set(ELEMENTAL_HYBRID ON)
  endif()

  # TODO: make nice
  set(MATH_LIBS "-L/opt/intel-16.0/linux/mkl/lib/intel64 -lmkl_rt")

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
      -D BUILD_SHARED_LIBS=ON
      -D EL_HYBRID=${ELEMENTAL_HYBRID}
      -D INSTALL_PYTHON_PACKAGE=${ELEMENTAL_INSTALL_PYTHON_PACKAGE}
  )

  # Get header files
  ExternalProject_Get_Property(project_Elemental install_dir)
  set(ELEMENTAL_INCLUDE_DIRS "${install_dir}/include")

  # Get library
  add_library(libelemental SHARED IMPORTED)
  set(ELEMENTAL_LIBRARIES "${install_dir}/lib/libelemental.so")
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
