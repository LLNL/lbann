# Defines the following variables:
#   - CNPY_FOUND
#   - CNPY_LIBRARIES
#   - CNPY_INCLUDE_DIRS
#
# Also creates an imported target CNPY

# Find the header
find_path(CNPY_INCLUDE_DIRS cnpy.h
  HINTS ${CNPY_DIR} $ENV{CNPY_DIR}
  PATH_SUFFIXES include
  NO_DEFAULT_PATH
  DOC "Directory with CNPY header.")
find_path(CNPY_INCLUDE_DIRS cnpy.h)

# Find the library
find_library(CNPY_LIBRARY cnpy
  HINTS ${CNPY_DIR} $ENV{CNPY_DIR}
  PATH_SUFFIXES lib lib64
  NO_DEFAULT_PATH
  DOC "The CNPY library.")
find_library(CNPY_LIBRARY cnpy)

# Setup the imported target
if (NOT TARGET CNPY)
  # Check if we have shared or static libraries
  include(LBANNCMakeUtilities)
  lbann_determine_library_type(${CNPY_LIBRARY} CNPY_LIB_TYPE)

  add_library(CNPY ${CNPY_LIB_TYPE} IMPORTED)
endif (NOT TARGET CNPY)

# Set the location
set_property(TARGET CNPY
  PROPERTY IMPORTED_LOCATION ${CNPY_LIBRARY})

# Set the include directories for the target
set_property(TARGET CNPY APPEND
  PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CNPY_INCLUDE_DIRS})

#
# Cleanup
#

# Set the include directories
set(CNPY_INCLUDE_DIRS ${CNPY_INCLUDE_DIRS}
  CACHE PATH
  "Directories in which to find headers for CNPY.")
mark_as_advanced(FORCE CNPY_INCLUDE_DIRS)

# Set the libraries
set(CNPY_LIBRARIES CNPY)
mark_as_advanced(FORCE CNPY_LIBRARY)

# Standard handling of the package arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CNPY
  DEFAULT_MSG
  CNPY_LIBRARY CNPY_INCLUDE_DIRS)
