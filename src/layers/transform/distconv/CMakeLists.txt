# Add the headers for this directory
set_full_path(THIS_DIR_CU_SOURCES
  distconv_gather_layer.cu
  distconv_scatter_layer.cu
  )

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
set(GPU_SOURCES "${GPU_SOURCES}" "${THIS_DIR_CU_SOURCES}" PARENT_SCOPE)