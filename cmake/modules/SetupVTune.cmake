# Exports the following variables
#
#   LBANN_HAS_VTUNE
#   VTUNE_INCLUDE_DIR
#   VTUNE_STATIC_LIB
#

find_path(VTUNE_INCLUDE_PATH libittnotify.h
  HINTS ${VTUNE_DIR} $ENV{VTUNE_DIR}
  PATH_SUFFIXES include
  NO_DEFAULT_PATH
  DOC "The location of VTune headers."
  )
#include_directories(${VTUNE_INCLUDE_PATH})

find_library(VTUNE_STATIC_LIB libittnotify.a 
  HINTS ${VTUNE_DIR} $ENV{VTUNE_DIR}
  PATH_SUFFIXES lib lib64 
  NO_DEFAULT_PATH
  DOC "The location of VTune Static lib"
  )


if(VTUNE_INCLUDE_PATH AND VTUNE_STATIC_LIB)
  include_directories(${VTUNE_INCLUDE_PATH})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DLBANN_VTUNE")
  set(LBANN_HAS_VTUNE TRUE)
endif()

#if(VTUNE_DIR)
#  message(STATUS "Found VTune: ${VTUNE_DIR}")
#
#  # Include directory.
#  set(VTUNE_INCLUDE_DIRS ${VTUNE_DIR}/include)
#  include_directories(${VTUNE_INCLUDE_DIRS})
#
#  # Set VTune static library.
#  set(VTUNE_STATIC_LIB ${VTUNE_DIR}/lib/libittnotify.a)
#
#  # Set preprocessor flag.
#  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DLBANN_VTUNE")
#
#  # LBANN has access to VTune.
#  set(LBANN_HAS_VTUNE TRUE)
#
#endif()
