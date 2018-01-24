# Add the headers for this directory
set_full_path(THIS_DIR_HEADERS
  reconstruction.hpp
  target_layer.hpp
  target_layer_distributed_minibatch.hpp
  target_layer_partitioned_minibatch.hpp
  )

# Propagate the files up the tree
set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
