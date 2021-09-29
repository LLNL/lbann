# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  checkpoint_binary.cpp
  checkpoint_file.cpp
  meta_learning_strategy.cpp
  mutation_strategy.cpp
  random_pairwise_exchange.cpp
  sendrecv_weights.cpp
  truncation_selection_exchange.cpp
  )

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
