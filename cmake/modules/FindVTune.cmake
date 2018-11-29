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
  DOC "The location of VTune headers.")
find_path(VTUNE_INCLUDE_PATH libittnotify.h)

find_library(VTUNE_STATIC_LIB libittnotify.a
  HINTS ${VTUNE_DIR} $ENV{VTUNE_DIR}
  PATH_SUFFIXES lib64 lib
  NO_DEFAULT_PATH
  DOC "The location of VTune Static library.")
find_library(VTUNE_STATIC_LIB libittnotify.a)

# Standard handling of the package arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(VTune
  REQUIRED_VARS VTUNE_INCLUDE_PATH VTUNE_STATIC_LIB)

if (VTUNE_INCLUDE_PATH AND VTUNE_STATIC_LIB)
  if (NOT TARGET vtune::vtune)
    add_library(vtune::vtune STATIC IMPORTED)
  endif ()

  set_target_properties(vtune::vtune PROPERTIES
    IMPORTED_LOCATION "${VTUNE_STATIC_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${VTUNE_INCLUDE_PATH}")

  set(VTUNE_LIBRARIES vtune::vtune)
endif ()
