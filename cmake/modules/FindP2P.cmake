# Defines the following variables:
#   - P2P_FOUND
#   - P2P_LIBRARIES
#   - P2P_INCLUDE_DIRS
#
# Also creates an imported target P2P

message(STATUS "P2P_DIR: ${P2P_DIR}")

# Find the header
find_path(P2P_INCLUDE_DIRS p2p/p2p.hpp
  HINTS ${P2P_DIR} $ENV{P2P_DIR}
  PATH_SUFFIXES include
  NO_DEFAULT_PATH
  DOC "Directory with P2P header.")
find_path(P2P_INCLUDE_DIRS p2p.hpp)

# Find the library
find_library(P2P_LIBRARY p2p
  HINTS ${P2P_DIR} $ENV{P2P_DIR}
  PATH_SUFFIXES lib64 lib
  NO_DEFAULT_PATH
  DOC "The P2P library.")
find_library(P2P_LIBRARY p2p)

# Standard handling of the package arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(P2P
  DEFAULT_MSG
  P2P_LIBRARY P2P_INCLUDE_DIRS)

# Setup the imported target
if (NOT TARGET P2P::P2P)
  add_library(P2P::P2P INTERFACE IMPORTED)
endif (NOT TARGET P2P::P2P)

# Set the include directories for the target
set_property(TARGET P2P::P2P APPEND
  PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${P2P_INCLUDE_DIRS})

# Set the link libraries for the target
set_property(TARGET P2P::P2P APPEND
  PROPERTY INTERFACE_LINK_LIBRARIES ${P2P_LIBRARY})

#
# Cleanup
#

# Set the include directories
mark_as_advanced(FORCE P2P_INCLUDE_DIRS)

# Set the libraries
set(P2P_LIBRARIES P2P::P2P)
mark_as_advanced(FORCE P2P_LIBRARY)
