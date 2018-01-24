# Add the headers for this directory
set_full_path(THIS_DIR_HEADERS
  greedy_layerwise_autoencoder.hpp
  model.hpp
  model_dag.hpp
  model_sequential.hpp
  siamese.hpp
  )

# Propagate the files up the tree
set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
