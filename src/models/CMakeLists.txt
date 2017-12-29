# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  directed_acyclic_graph.cpp
  recurrent.cpp
  greedy_layerwise_autoencoder.cpp
  model.cpp
  sequential.cpp
  siamese.cpp
  )

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
