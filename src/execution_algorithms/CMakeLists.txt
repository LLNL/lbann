# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  batch_functional_inference_algorithm.cpp
  sgd_training_algorithm.cpp
  training_algorithm.cpp
  )

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
