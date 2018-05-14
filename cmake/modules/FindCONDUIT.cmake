# Defines the following variables:
#   - CONDUIT_FOUND
#   - CONDUIT_LIBRARIES
#   - CONDUIT_INCLUDE_DIRS
#
# Also creates an imported target CONDUIT

# Find the header
find_path(CONDUIT_INCLUDE_DIRS conduit/conduit_relay.h
  HINTS ${LBANN_CONDUIT_DIR} ${CONDUIT_DIR} $ENV{CONDUIT_DIR}
  PATH_SUFFIXES include
  NO_DEFAULT_PATH
  DOC "Directory with CONDUIT header.")
find_path(CONDUIT_INCLUDE_DIRS conduit/conduit_relay.h)

# Find the library
find_library(CONDUIT_RELAY_LIBRARY conduit_relay
  HINTS ${LBANN_CONDUIT_DIR} ${CONDUIT_DIR} $ENV{CONDUIT_DIR}
  PATH_SUFFIXES lib64 lib
  NO_DEFAULT_PATH
  DOC "The CONDUIT_RELAY library.")
find_library(CONDUIT_RELAY_LIBRARY conduit_relay)

# Find the library
find_library(CONDUIT_LIBRARY conduit
  HINTS ${LBANN_CONDUIT_DIR} ${CONDUIT_DIR} $ENV{CONDUIT_DIR}
  PATH_SUFFIXES lib64 lib
  NO_DEFAULT_PATH
  DOC "The CONDUIT library.")
find_library(CONDUIT_LIBRARY conduit)

# Standard handling of the package arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CONDUIT
  DEFAULT_MSG
  CONDUIT_RELAY_LIBRARY CONDUIT_LIBRARY CONDUIT_INCLUDE_DIRS)

# Setup the imported target
if (NOT TARGET CONDUIT::CONDUIT)
  add_library(CONDUIT::CONDUIT INTERFACE IMPORTED)
endif (NOT TARGET CONDUIT::CONDUIT)

# Set the include directories for the target
set_property(TARGET CONDUIT::CONDUIT APPEND
  PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CONDUIT_INCLUDE_DIRS})

# Set the link libraries for the target
set_property(TARGET CONDUIT::CONDUIT APPEND
  PROPERTY INTERFACE_LINK_LIBRARIES ${CONDUIT_RELAY_LIBRARY})
set_property(TARGET CONDUIT::CONDUIT APPEND
  PROPERTY INTERFACE_LINK_LIBRARIES ${CONDUIT_LIBRARY})

#
# Cleanup
#

# Set the include directories
mark_as_advanced(FORCE CONDUIT_INCLUDE_DIRS)

# Set the libraries
set(CONDUIT_LIBRARIES CONDUIT::CONDUIT)
mark_as_advanced(FORCE CONDUIT_RELAY_LIBRARY)
mark_as_advanced(FORCE CONDUIT_LIBRARY)
