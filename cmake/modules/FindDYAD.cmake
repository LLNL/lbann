# Defines the following variables:
#   - DYAD_FOUND
#   - DYAD_LIBRARIES
#   - DYAD_INCLUDE_DIRS
#
# Also creates an imported target DYAD

# Find the header
find_path(DYAD_INCLUDE_DIRS dyad_stream_api.hpp
  HINTS ${DYAD_DIR} $ENV{DYAD_DIR}
  PATH_SUFFIXES include
  NO_DEFAULT_PATH
  DOC "Directory with DYAD header.")
find_path(DYAD_INCLUDE_DIRS dyad_stream_api.hpp)

# Find the library
find_library(DYAD_LIBRARY dyad_fstream
  HINTS ${DYAD_DIR} $ENV{DYAD_DIR}
  PATH_SUFFIXES lib64 lib
  NO_DEFAULT_PATH
  DOC "The DYAD library.")
find_library(DYAD_LIBRARY dyad_fstream)

# Standard handling of the package arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(DYAD
  DEFAULT_MSG
  DYAD_LIBRARY DYAD_INCLUDE_DIRS)

# Setup the imported target
if (NOT TARGET DYAD::DYAD)
  add_library(DYAD::DYAD INTERFACE IMPORTED)
endif (NOT TARGET DYAD::DYAD)

# Set the include directories for the target
set_property(TARGET DYAD::DYAD APPEND
  PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${DYAD_INCLUDE_DIRS})

# Set the link libraries for the target
set_property(TARGET DYAD::DYAD APPEND
  PROPERTY INTERFACE_LINK_LIBRARIES ${DYAD_LIBRARY})

#
# Cleanup
#

# Set the include directories
mark_as_advanced(FORCE DYAD_INCLUDE_DIRS)

# Set the libraries
set(DYAD_LIBRARIES DYAD::DYAD)
mark_as_advanced(FORCE DYAD_LIBRARY)
