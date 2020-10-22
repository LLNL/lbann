# Output variables
#
#   NVSHMEM_FOUND
#   NVSHMEM_LIBRARY
#   NVSHMEM_INCLUDE_DIRS
#
# Also creates an imported target NVSHMEM::NVSHMEM

# Find the library
find_library(NVSHMEM_LIBRARY nvshmem
  HINTS ${NVSHMEM_DIR} $ENV{NVSHMEM_DIR}
  PATH_SUFFIXES lib lib64
  NO_DEFAULT_PATH
  DOC "The location of NVSHMEM library.")
find_library(NVSHMEM_LIBRARY nvshmem)

# Find the header
find_path(NVSHMEM_INCLUDE_DIRS nvshmem.h
  HINTS ${NVSHMEM_DIR} $ENV{NVSHMEM_DIR}
  PATH_SUFFIXES include
  NO_DEFAULT_PATH
  DOC "The location of NVSHMEM headers.")
find_path(NVSHMEM_INCLUDE_DIRS nvshmemx.h)

# Handle the find_package arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  NVSHMEM DEFAULT_MSG NVSHMEM_LIBRARY NVSHMEM_INCLUDE_DIRS)

# Build the imported target
if (NOT TARGET NVSHMEM::NVSHMEM)
  add_library(NVSHMEM::NVSHMEM INTERFACE IMPORTED)
  set_property(TARGET NVSHMEM::NVSHMEM PROPERTY
    INTERFACE_LINK_LIBRARIES ${NVSHMEM_LIBRARY})
  set_property(TARGET NVSHMEM::NVSHMEM PROPERTY
    INTERFACE_INCLUDE_DIRECTORIES ${NVSHMEM_INCLUDE_DIRS})
endif ()

if (NVSHMEM_FOUND)
  # Workaround for separable compilation with cooperative threading. see
  # https://stackoverflow.com/questions/53492528/cooperative-groupsthis-grid-causes-any-cuda-api-call-to-return-unknown-erro.
  # Adding this to INTERFACE_COMPILE_OPTIONS does not seem to solve the problem.
  # It seems that CMake does not add necessary options for device linking when cuda_add_executable/library is NOT used. See also
  # https://github.com/dealii/dealii/pull/5405
  string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_70,code=compute_70")
endif ()
