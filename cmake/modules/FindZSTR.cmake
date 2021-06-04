# Defines the following variables:
#   - ZSTR_FOUND
#   - ZSTR_INCLUDE_DIRS
#
# Also creates an imported target ZSTR::ZSTR
#
# I can't find any interesting version information in the headers. So
# good luck with that.

# Find the header
find_path(ZSTR_INCLUDE_DIR zstr.hpp
  HINTS ${ZSTR_DIR} $ENV{ZSTR_DIR}
  PATH_SUFFIXES include
  NO_DEFAULT_PATH
  DOC "Directory with ZSTR header.")
find_path(ZSTR_INCLUDE_DIR zstr.hpp)

set(ZSTR_INCLUDE_DIRS "${ZSTR_INCLUDE_DIR}"
  CACHE STRING "The list of paths required for ZSTR usage" FORCE)

# Standard handling of the package arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ZSTR
  DEFAULT_MSG
  ZSTR_INCLUDE_DIR)

# Setup the imported target
if (NOT TARGET ZSTR::ZSTR)
  add_library(ZSTR::ZSTR INTERFACE IMPORTED)
endif (NOT TARGET ZSTR::ZSTR)

# Set the include directories for the target
set_property(TARGET ZSTR::ZSTR APPEND
  PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${ZSTR_INCLUDE_DIRS})

#
# Cleanup
#

# Set the include directories
mark_as_advanced(FORCE ZSTR_INCLUDE_DIRS)
