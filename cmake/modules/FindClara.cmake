# Output variables
#
#   Clara_FOUND
#   Clara_LIBRARIES
#   Clara_INCLUDE_PATH
#
# Also creates an imported target clara::clara

# Find the header
find_path(CLARA_INCLUDE_PATH clara.hpp
  HINTS ${CLARA_DIR} $ENV{CLARA_DIR} ${Clara_DIR} $ENV{Clara_DIR}
  PATH_SUFFIXES include
  NO_DEFAULT_PATH)
find_path(CLARA_INCLUDE_PATH clara.hpp)

# Handle the find_package arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  Clara DEFAULT_MSG CLARA_INCLUDE_PATH)

# Build the imported target
if (NOT TARGET clara::clara)
  add_library(clara::clara INTERFACE IMPORTED)
endif()

set_property(TARGET clara::clara
  PROPERTY INTERFACE_INCLUDE_DIRECTORIES
  ${CLARA_INCLUDE_PATH})

# Set the last of the output variables
set(CLARA_LIBRARIES clara::clara)

# Cleanup
mark_as_advanced(FORCE CLARA_INCLUDE_PATH)
