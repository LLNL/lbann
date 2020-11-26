# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  batch_normalization.cpp
  entrywise_batch_normalization.cpp
  instance_norm.cpp
  layer_norm.cpp
  )

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
