# Defines the following variables:
#   - DIHYDROGEN_FOUND
#   - DIHYDROGEN_LIBRARIES
#   - DIHYDROGEN_INCLUDE_DIRS
#
# Also creates an imported target DIHYDORGEN

message(STATUS "DIHYDROGEN_DIR: ${DIHYDROGEN_DIR}")

# Find the header
find_path(DIHYDROGEN_INCLUDE_DIRS distconv/distconv.hpp
  HINTS ${DIHYDROGEN_DIR} $ENV{DIHYDROGEN_DIR}
  PATH_SUFFIXES include
  NO_DEFAULT_PATH
  DOC "Directory with DIHYDROGEN header.")
find_path(DIHYDROGEN_INCLUDE_DIRS distconv.hpp)

# Find the library
find_library(DIHYDROGEN_LIBRARY distconv
  HINTS ${DIHYDROGEN_DIR} $ENV{DIHYDROGEN_DIR}
  PATH_SUFFIXES lib64 lib
  NO_DEFAULT_PATH
  DOC "The DIHYDROGEN library.")
find_library(DIHYDROGEN_LIBRARY distconv)

# Standard handling of the package arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(DIHYDROGEN
  DEFAULT_MSG
  DIHYDROGEN_LIBRARY DIHYDROGEN_INCLUDE_DIRS)

# Setup the imported target
if (NOT TARGET DIHYDROGEN::DIHYDROGEN)
  add_library(DIHYDROGEN::DIHYDROGEN INTERFACE IMPORTED)
endif (NOT TARGET DIHYDROGEN::DIHYDROGEN)

# Set the include directories for the target
set_property(TARGET DIHYDROGEN::DIHYDROGEN APPEND
  PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${DIHYDROGEN_INCLUDE_DIRS})

# Set the link libraries for the target
set_property(TARGET DIHYDROGEN::DIHYDROGEN APPEND
  PROPERTY INTERFACE_LINK_LIBRARIES ${DIHYDROGEN_LIBRARY})

#
# Cleanup
#

# Set the include directories
mark_as_advanced(FORCE DIHYDROGEN_INCLUDE_DIRS)

# Set the libraries
set(DIHYDROGEN_LIBRARIES DIHYDROGEN::DIHYDROGEN)
mark_as_advanced(FORCE DIHYDROGEN_LIBRARY)
