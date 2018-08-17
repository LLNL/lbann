# Defines the following variables:
#   - DISTCONV_FOUND
#   - DISTCONV_LIBRARIES
#   - DISTCONV_INCLUDE_DIRS
#
# Also creates an imported target DISTCONV

message(STATUS "DISTCONV_DIR: ${DISTCONV_DIR}")

# Find the header
find_path(DISTCONV_INCLUDE_DIRS distconv/distconv.hpp
  HINTS ${DISTCONV_DIR} $ENV{DISTCONV_DIR}
  PATH_SUFFIXES include
  NO_DEFAULT_PATH
  DOC "Directory with DISTCONV header.")
find_path(DISTCONV_INCLUDE_DIRS distconv.hpp)

# Find the library
find_library(DISTCONV_LIBRARY distconv
  HINTS ${DISTCONV_DIR} $ENV{DISTCONV_DIR}
  PATH_SUFFIXES lib64 lib
  NO_DEFAULT_PATH
  DOC "The DISTCONV library.")
find_library(DISTCONV_LIBRARY distconv)

# Standard handling of the package arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(DISTCONV
  DEFAULT_MSG
  DISTCONV_LIBRARY DISTCONV_INCLUDE_DIRS)

# Setup the imported target
if (NOT TARGET DISTCONV::DISTCONV)
  add_library(DISTCONV::DISTCONV INTERFACE IMPORTED)
endif (NOT TARGET DISTCONV::DISTCONV)

# Set the include directories for the target
set_property(TARGET DISTCONV::DISTCONV APPEND
  PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${DISTCONV_INCLUDE_DIRS})

# Set the link libraries for the target
set_property(TARGET DISTCONV::DISTCONV APPEND
  PROPERTY INTERFACE_LINK_LIBRARIES ${DISTCONV_LIBRARY})

#
# Cleanup
#

# Set the include directories
mark_as_advanced(FORCE DISTCONV_INCLUDE_DIRS)

# Set the libraries
set(DISTCONV_LIBRARIES DISTCONV::DISTCONV)
mark_as_advanced(FORCE DISTCONV_LIBRARY)
