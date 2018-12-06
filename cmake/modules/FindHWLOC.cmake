# Output variables
#
#   HWLOC_FOUND
#   HWLOC_LIBRARIES
#   HWLOC_INCLUDE_PATH
#
# Also creates an imported target HWLOC::hwloc

if (MPI_FOUND)
  list(APPEND _TMP_MPI_LIBS "${MPI_C_LIBRARIES}" "${MPI_CXX_LIBRARIES}")
  foreach (lib IN LISTS _TMP_MPI_LIBS)
    get_filename_component(_TMP_MPI_LIB_DIR "${lib}" DIRECTORY)
    list(APPEND _TMP_MPI_LIBRARY_DIRS ${_TMP_MPI_LIB_DIR})
  endforeach ()

  if (_TMP_MPI_LIBRARY_DIRS)
    list(REMOVE_DUPLICATES _TMP_MPI_LIBRARY_DIRS)
  endif ()
endif (MPI_FOUND)

# Find the library
find_library(HWLOC_LIBRARY hwloc
  HINTS ${HWLOC_DIR} $ENV{HWLOC_DIR} ${_TMP_MPI_LIBRARY_DIRS}
  PATH_SUFFIXES lib64 lib
  NO_DEFAULT_PATH)
find_library(HWLOC_LIBRARY hwloc)

# Find the header
find_path(HWLOC_INCLUDE_PATH hwloc.h
  HINTS ${HWLOC_DIR} $ENV{HWLOC_DIR}
  ${MPI_C_INCLUDE_PATH} ${MPI_CXX_INCLUDE_PATH}
  PATH_SUFFIXES include
  NO_DEFAULT_PATH)
find_path(HWLOC_INCLUDE_PATH hwloc.h)

# Handle the find_package arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  HWLOC DEFAULT_MSG HWLOC_LIBRARY HWLOC_INCLUDE_PATH)

# Build the imported target
if (NOT TARGET HWLOC::hwloc)
  add_library(HWLOC::hwloc INTERFACE IMPORTED)
endif()

set_property(TARGET HWLOC::hwloc
  PROPERTY INTERFACE_LINK_LIBRARIES ${HWLOC_LIBRARY})

if (NOT "/usr/include" STREQUAL "${HWLOC_INCLUDE_PATH}")
  set_property(TARGET HWLOC::hwloc
    PROPERTY INTERFACE_INCLUDE_DIRECTORIES
    ${HWLOC_INCLUDE_PATH})
endif ()

# Set the last of the output variables
set(HWLOC_LIBRARIES HWLOC::hwloc)

# Cleanup
mark_as_advanced(FORCE HWLOC_INCLUDE_PATH)
mark_as_advanced(FORCE HWLOC_LIBRARY)
