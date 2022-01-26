# Add the headers for this directory
set_full_path(THIS_DIR_SOURCES
  gather_layer.cu
  )

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)