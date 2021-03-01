# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  batch_normalization.cpp
  dropout.cpp
  entrywise_batch_normalization.cpp
  instance_norm.cpp
  layer_norm.cpp
  local_response_normalization.cpp
  selu_dropout.cpp
  )

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
