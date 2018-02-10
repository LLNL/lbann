# Defines the following variables:
#   - ALUMINUM_FOUND
#   - ALUMINUM_LIBRARIES
#   - ALUMINUM_INCLUDE_DIRS
#
# Also creates an imported target ALUMINUM

# Find the header
find_path(ALUMINUM_INCLUDE_DIRS allreduce.hpp
  HINTS ${ALUMINUM_DIR} $ENV{ALUMINUM_DIR}
  PATH_SUFFIXES include src
  NO_DEFAULT_PATH
  DOC "Directory with ALUMINUM header.")
find_path(ALUMINUM_INCLUDE_DIRS allreduce.hpp)

# Find the library
find_library(ALUMINUM_LIBRARY allreduce
  HINTS ${ALUMINUM_DIR} $ENV{ALUMINUM_DIR}
  PATH_SUFFIXES lib64 lib
  NO_DEFAULT_PATH
  DOC "The ALUMINUM library.")
find_library(ALUMINUM_LIBRARY allreduce)

# Standard handling of the package arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ALUMINUM
  DEFAULT_MSG
  ALUMINUM_LIBRARY ALUMINUM_INCLUDE_DIRS)

# Setup the imported target
if (NOT TARGET ALUMINUM::ALUMINUM)
  add_library(ALUMINUM::ALUMINUM INTERFACE IMPORTED)
endif (NOT TARGET ALUMINUM::ALUMINUM)

# Set the include directories for the target
set_property(TARGET ALUMINUM::ALUMINUM APPEND
  PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${ALUMINUM_INCLUDE_DIRS})

# Set the link libraries for the target
set_property(TARGET ALUMINUM::ALUMINUM APPEND
  PROPERTY INTERFACE_LINK_LIBRARIES ${ALUMINUM_LIBRARY})

#
# Cleanup
#

# Set the include directories
mark_as_advanced(FORCE ALUMINUM_INCLUDE_DIRS)

# Set the libraries
set(ALUMINUM_LIBRARIES ALUMINUM::ALUMINUM)
mark_as_advanced(FORCE ALUMINUM_LIBRARY)
