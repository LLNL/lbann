# Add the headers for this directory
if (LBANN_HAS_DISTCONV)
if (LBANN_HAS_NVSHMEM)
set_full_path(THIS_DIR_SOURCES
  distconv_gather.cpp
  distconv_scatter.cpp
)
endif()
endif()
# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
set(GPU_SOURCES "${GPU_SOURCES}" "${THIS_DIR_CU_SOURCES}" PARENT_SCOPE)