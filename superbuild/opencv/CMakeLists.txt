enable_language(C)
enable_language(CXX)
enable_language(Fortran)# FIXME: Really??

# Add all the OpenCV options; this is just for the GUI, of course
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
include(AddOpenCVOptions)

set(OPENCV_CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}"
  CACHE PATH "The installation destination of OpenCV")

set(OPENCV_CMAKE_BUILD_TYPE "${CMAKE_BUILD_TYPE}"
  CACHE STRING "The build type for OpenCV.")

set(OPENCV_CMAKE_GENERATOR "${CMAKE_GENERATOR}"
  CACHE STRING "The generator used by CMake for OpenCV.")

if (TARGET JPEG-TURBO)
  set(_opencv_depends_tag DEPENDS)
  list(APPEND _OPENCV_DEPENDS JPEG-TURBO)
  set(OPENCV_WITH_LIBJPEG_TURBO ON)
  set(OPENCV_BUILD_JPEG OFF)
  
  set(OPENCV_CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH}" "${JPEG-TURBO_CMAKE_INSTALL_PREFIX}") 
  #set(OPENCV_JPEG_INCLUDE_DIR "${CMAKE_INSTALL_PREFIX}/include"
  #    CACHE STRING "The include path to jpeg-turbo for OpenCv.")
  #  set(OPENCV_JPEG_LIBRARY "${CMAKE_INSTALL_PREFIX}/lib"
  #  CACHE STRING "The library path to jpeg-turbo for OpenCv.")
endif ()


# Get the list of opencv variables
get_property(OPENCV_VARIABLES DIRECTORY PROPERTY VARIABLES)
list(FILTER OPENCV_VARIABLES INCLUDE REGEX "^OPENCV_.*")

create_cmake_arguments(
  OUTPUT_VARIABLE OPENCV_CMAKE_ARGS
  PACKAGE_NAME OPENCV
  REMOVE_PKG_NAME
  VARIABLES ${OPENCV_VARIABLES})

set(OPENCV_URL "https://github.com/opencv/opencv"
  CACHE STRING "The URL from which to clone OpenCV")

set(OPENCV_TAG "3.3.0"
  CACHE STRING "The git tag or hash to checkout for OpenCV")

include(ExternalProject)
ExternalProject_Add(OPENCV
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}
  TMP_DIR ${CMAKE_CURRENT_BINARY_DIR}/tmp
  STAMP_DIR ${CMAKE_CURRENT_BINARY_DIR}/stamp
  GIT_REPOSITORY ${OPENCV_URL}
  GIT_TAG ${OPENCV_TAG}
  ${_opencv_depends_tag} ${_OPENCV_DEPENDS}
  SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/src
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/build
  INSTALL_DIR ${OPENCV_CMAKE_INSTALL_PREFIX}
  USES_TERMINAL_BUILD 1
  LIST_SEPARATOR |
  CMAKE_ARGS
  -G${OPENCV_CMAKE_GENERATOR}
  -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
  -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
  ${OPENCV_CMAKE_ARGS}
  )

set(OPENCV_DIR ${OPENCV_CMAKE_INSTALL_PREFIX}
  CACHE INTERNAL "The install prefix of OPENCV.")
