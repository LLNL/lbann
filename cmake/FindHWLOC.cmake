# Output variables
#
#   HWLOC_FOUND
#   HWLOC_LIBRARIES
#   HWLOC_INCLUDE_PATH
#
# Also creates an imported target HWLOC::hwloc

# Find the library
find_library(HWLOC_LIBRARY hwloc
  HINTS ${HWLOC_DIR} $ENV{HWLOC_DIR}
  PATH_SUFFIXES lib64 lib
  NO_DEFAULT_PATH)
find_library(HWLOC_LIBRARY hwloc)

# Find the header
find_path(HWLOC_INCLUDE_PATH hwloc.h
  HINTS ${HWLOC_DIR} $ENV{HWLOC_DIR}
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

set_property(TARGET HWLOC::hwloc
  PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${HWLOC_INCLUDE_PATH})

# Set the last of the output variables
set(HWLOC_LIBRARIES HWLOC::hwloc)
