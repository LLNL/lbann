# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  checkpoint_binary.cpp
  checkpoint_file.cpp
  checkpoint_file_dyad.cpp
  meta_learning_strategy.cpp
  random_pairwise_exchange.cpp
  sendrecv_weights.cpp
  )

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
