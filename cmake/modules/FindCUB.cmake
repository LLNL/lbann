# Exports the following variables
#
#   CUB_FOUND
#   CUB_INCLUDE_PATH
#   CUB_LIBRARIES
#
# Also adds the following imported target:
#
#   cuda::cub
#

find_path(CUB_INCLUDE_PATH cub/cub.cuh
  HINTS ${CUB_DIR} $ENV{CUB_DIR} ${CUDA_TOOLKIT_ROOT_DIR} ${CUDA_SDK_ROOT_DIR}
  PATH_SUFFIXES include
  NO_DEFAULT_PATH
  DOC "The CUB header directory."
  )
find_path(CUB_INCLUDE_PATH cub/cub.cuh)

# Standard handling of the package arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CUB
  DEFAULT_MSG CUB_INCLUDE_PATH)

# Setup the imported target
if (NOT TARGET cuda::cub)
  add_library(cuda::cub INTERFACE IMPORTED)
endif (NOT TARGET cuda::cub)

# Set the include directories for the target
set_property(TARGET cuda::cub APPEND
  PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CUB_INCLUDE_PATH})

#
# Cleanup
#

# Set the include directories
mark_as_advanced(FORCE CUB_INCLUDE_DIRS)

# Set the libraries
set(CUB_LIBRARIES cuda::cub)
