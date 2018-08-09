# Defines the following variables:
#   - CONDUIT_FOUND
#   - CONDUIT_LIBRARIES
#   - CONDUIT_INCLUDE_DIRS
#

find_file(CONDUIT_CMAKE_FILE conduit.cmake
  HINTS ${LBANN_CONDUIT_DIR} ${CONDUIT_DIR} $ENV{CONDUIT_DIR}
  PATH_SUFFIXES lib/cmake lib/cmake/conduit
  NO_DEFAULT_PATH
  DOC "CONDUIT target file.")
find_file(CONDUIT_CMAKE_FILE conduit.cmake)

find_path(CONDUIT_INCLUDE_DIRS conduit/conduit_relay.h
  HINTS ${LBANN_CONDUIT_DIR} ${CONDUIT_DIR} $ENV{CONDUIT_DIR}
  PATH_SUFFIXES include
  NO_DEFAULT_PATH
  DOC "Directory with CONDUIT header.")
find_path(CONDUIT_INCLUDE_DIRS conduit/conduit_relay.h)

if (CONDUIT_CMAKE_FILE AND NOT TARGET conduit)
  include("${CONDUIT_CMAKE_FILE}")
endif ()

# Standard handling of the package arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CONDUIT
  DEFAULT_MSG CONDUIT_CMAKE_FILE)

# Setup the imported target
if (NOT TARGET CONDUIT::CONDUIT)
  add_library(CONDUIT::CONDUIT INTERFACE IMPORTED)
endif (NOT TARGET CONDUIT::CONDUIT)

# Set the link libraries for the target
set_property(TARGET CONDUIT::CONDUIT APPEND
  PROPERTY INTERFACE_LINK_LIBRARIES conduit conduit_relay conduit_blueprint)

set_property(TARGET CONDUIT::CONDUIT APPEND
  PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${CONDUIT_INCLUDE_DIRS}")

#
# Cleanup
#

# Set the include directories
mark_as_advanced(FORCE CONDUIT_CMAKE_FILE)

# Set the libraries
set(CONDUIT_LIBRARIES CONDUIT::CONDUIT)
