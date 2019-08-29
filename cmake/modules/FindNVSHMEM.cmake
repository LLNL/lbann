find_path(NVSHMEM_INCLUDE_PATH 
  NAMES nvshmem.h nvshmemx.h shmem.h shmemx.h
  HINTS ${NVSHMEM_DIR} $ENV{NVSHMEM_DIR}
  PATH_SUFFIXES include include/nvprefix
  NO_DEFAULT_PATH
  DOC "The NVSHMEM include path.")

find_library(NVSHMEM_LIBRARY 
  NAMES nvshmem shmem
  HINTS ${NVSHMEM_DIR} $ENV{NVSHMEM_DIR}
  PATH_SUFFIXES lib64 lib lib/nvprefix lib64/nvprefix
  NO_DEFAULT_PATH
  DOC "The NVSHMEM library.")

# Standard handling of the package arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NVSHMEM
  REQUIRED_VARS NVSHMEM_LIBRARY NVSHMEM_INCLUDE_PATH)

if (NOT TARGET cuda::nvshmem)
  add_library(cuda::nvshmem INTERFACE IMPORTED)
endif ()

if (NVSHMEM_INCLUDE_PATH AND NVSHMEM_LIBRARY)
  set_target_properties(cuda::nvshmem PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${NVSHMEM_INCLUDE_PATH}"
    INTERFACE_LINK_LIBRARIES "${NVSHMEM_LIBRARY}")
endif ()

set(NVSHMEM_LIBRARIES cuda::nvshmem)
mark_as_advanced(NVSHMEM_INCLUDE_PATH)
mark_as_advanced(NVSHMEM_LIBRARY)
