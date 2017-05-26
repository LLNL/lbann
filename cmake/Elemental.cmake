include(ExternalProject)

# Options
if(NOT ELEMENTAL_LIBRARY_TYPE)
  set(ELEMENTAL_LIBRARY_TYPE SHARED)
endif()
option(FORCE_ELEMENTAL_BUILD "Elemental: force build" OFF)

# Check if Elemental has been provided
if(Elemental_DIR AND NOT FORCE_ELEMENTAL_BUILD)

  # Status message
  message(STATUS "Found Elemental: ${Elemental_DIR}")

else()

  # Git repository URL and tag
  if(NOT ELEMENTAL_URL)
    set(ELEMENTAL_URL https://github.com/elemental/Elemental.git)
  endif()
  if(NOT ELEMENTAL_TAG)
     # Deprecated - Commit from 9/11/2016
     # Deprecated - set(ELEMENTAL_TAG "d14e8f396cbafac8cf6b46da442ad3b7a1d42508")
     # 0.87.6
     # set(ELEMENTAL_TAG "8a1a42c4391b73e1b7e9ea07736459cccb7d6b21")
     # 0.87.7
     set(ELEMENTAL_TAG "477e503a7a840cc1a75173552711b980505a0b06")
  endif()
  message(STATUS "Will pull Elemental (tag ${ELEMENTAL_TAG}) from ${ELEMENTAL_URL}")

  # Elemental build options
  if(NOT ELEMENTAL_BUILD_TYPE)
    set(ELEMENTAL_BUILD_TYPE ${CMAKE_BUILD_TYPE})
  endif()

  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    option(ELEMENTAL_HYBRID "Elemental: make use of OpenMP within MPI packing/unpacking" OFF)
  else()
   option(ELEMENTAL_HYBRID "Elemental: make use of OpenMP within MPI packing/unpacking" ON)
  endif()
  option(ELEMENTAL_C_INTERFACE "Elemental: build C interface?" OFF)
  option(ELEMENTAL_INSTALL_PYTHON_PACKAGE "Elemental: install Python interface?" OFF)
  option(ELEMENTAL_DISABLE_PARMETIS "Elemental: disable ParMETIS?" ON) # Non-commercial license
  option(ELEMENTAL_DISABLE_QUAD "Elemental: disable quad precision" ON) # GPL license
  option(ELEMENTAL_USE_64BIT_INTS "Elemental: use 64bit integers" ON)
  option(ELEMENTAL_USE_64BIT_BLAS_INTS "Elemental: use 64bit integers for BLAS" OFF)

  # Determine library type
  if(ELEMENTAL_LIBRARY_TYPE STREQUAL STATIC)
    set(ELEMENTAL_BUILD_SHARED_LIBS OFF)
  elseif(ELEMENTAL_LIBRARY_TYPE STREQUAL SHARED)
    set(ELEMENTAL_BUILD_SHARED_LIBS ON)
  else()
    message(WARNING "Elemental: unknown library type (${ELEMENTAL_LIBRARY_TYPE}), defaulting to shared library.")
    set(ELEMENTAL_BUILD_SHARED_LIBS ON)
  endif()

  # Download and build location
  set(ELEMENTAL_SOURCE_DIR ${PROJECT_BINARY_DIR}/download/elemental/source)
  set(ELEMENTAL_BINARY_DIR ${PROJECT_BINARY_DIR}/download/elemental/build)

  # patch file
  set(PATCH_DIR ${PROJECT_SOURCE_DIR}/external)
  set(EL_OpenBLAS_PATCH_DIR ${PATCH_DIR}/OpenBLAS)
  if (PATCH_OPENBLAS)
    set(EL_OpenBLAS_PATCH_SCRIPT ${EL_OpenBLAS_PATCH_DIR}/patchELOpenBLAS.sh)
  else()
    set(EL_OpenBLAS_PATCH_SCRIPT)
  endif()

  # Get Elemental from Git repository and build
  ExternalProject_Add(project_Elemental
    PREFIX          ${CMAKE_INSTALL_PREFIX}
    TMP_DIR         ${ELEMENTAL_BINARY_DIR}/tmp
    STAMP_DIR       ${ELEMENTAL_BINARY_DIR}/stamp
    GIT_REPOSITORY  ${ELEMENTAL_URL}
    GIT_TAG         ${ELEMENTAL_TAG}
    SOURCE_DIR      ${ELEMENTAL_SOURCE_DIR}
    BINARY_DIR      ${ELEMENTAL_BINARY_DIR}
    BUILD_COMMAND   pushd ${ELEMENTAL_SOURCE_DIR} && ${EL_OpenBLAS_PATCH_SCRIPT} && popd && ${CMAKE_MAKE_PROGRAM} -j${MAKE_NUM_PROCESSES} VERBOSE=${VERBOSE} CC=${MPI_C_COMPILER} CXX=${MPI_CXX_COMPILER}
    INSTALL_DIR     ${CMAKE_INSTALL_PREFIX}
    INSTALL_COMMAND ${CMAKE_MAKE_PROGRAM} install -j${MAKE_NUM_PROCESSES} VERBOSE=${VERBOSE}
    CMAKE_ARGS
      -D CMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
      -D CMAKE_INSTALL_MESSAGE=${CMAKE_INSTALL_MESSAGE}
      -D CMAKE_BUILD_TYPE=${ELEMENTAL_BUILD_TYPE}
      -D CMAKE_C_COMPILER=${CMAKE_C_COMPILER}
      -D CMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
      -D CMAKE_Fortran_COMPILER=${CMAKE_Fortran_COMPILER}
      -D GFORTRAN_LIB=${GFORTRAN_LIB}
      -D MPI_C_COMPILER=${MPI_C_COMPILER}
      -D MPI_CXX_COMPILER=${MPI_CXX_COMPILER}
      -D MPI_Fortran_COMPILER=${MPI_Fortran_COMPILER}
      -D MATH_LIBS=${ELEMENTAL_MATH_LIBS}
      -D BUILD_SHARED_LIBS=${ELEMENTAL_BUILD_SHARED_LIBS}
      -D EL_HYBRID=${ELEMENTAL_HYBRID}
      -D EL_C_INTERFACE=${ELEMENTAL_C_INTERFACE}
      -D INSTALL_PYTHON_PACKAGE=${ELEMENTAL_INSTALL_PYTHON_PACKAGE}
      -D EL_DISABLE_PARMETIS=${ELEMENTAL_DISABLE_PARMETIS}
      -D EL_DISABLE_QUAD=${ELEMENTAL_DISABLE_QUAD}
      -D EL_USE_64BIT_INTS=${ELEMENTAL_USE_64BIT_INTS}
      -D EL_USE_64BIT_BLAS_INTS=${ELEMENTAL_USE_64BIT_BLAS_INTS}
      -D CMAKE_SKIP_BUILD_RPATH=${CMAKE_SKIP_BUILD_RPATH}
      -D CMAKE_BUILD_WITH_INSTALL_RPATH=${CMAKE_BUILD_WITH_INSTALL_RPATH}
      -D CMAKE_INSTALL_RPATH_USE_LINK_PATH=${CMAKE_INSTALL_RPATH_USE_LINK_PATH}
      -D CMAKE_INSTALL_RPATH=${CMAKE_INSTALL_RPATH}
      -D CMAKE_MACOSX_RPATH=${CMAKE_MACOSX_RPATH}
      -D PATCH_DIR=${PATCH_DIR}
  )

  # Get install directory
  set(Elemental_DIR ${CMAKE_INSTALL_PREFIX})

  # LBANN has built Elemental
  set(LBANN_BUILT_ELEMENTAL TRUE)

endif()

# Include header files
set(Elemental_INCLUDE_DIRS ${Elemental_DIR}/${CMAKE_INSTALL_INCLUDEDIR})
include_directories(SYSTEM ${Elemental_INCLUDE_DIRS})

# Get library
if(ELEMENTAL_LIBRARY_TYPE STREQUAL STATIC)
  set(Elemental_LIBRARIES ${Elemental_DIR}/${CMAKE_INSTALL_LIBDIR}/${CMAKE_STATIC_LIBRARY_PREFIX}El${CMAKE_STATIC_LIBRARY_SUFFIX})
else()
  set(Elemental_LIBRARIES ${Elemental_DIR}/${CMAKE_INSTALL_LIBDIR}/${CMAKE_SHARED_LIBRARY_PREFIX}El${CMAKE_SHARED_LIBRARY_SUFFIX})
endif()

# Add preprocessor flag for Elemental
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__LIB_ELEMENTAL")

# LBANN has access to Elemental
set(LBANN_HAS_ELEMENTAL TRUE)
