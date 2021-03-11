# Add the headers for this directory
set_full_path(THIS_DIR_HEADERS
  batch_functional_inference_algorithm.hpp
  training_algorithm.hpp
  sgd_training_algorithm.hpp
  )

# Propagate the files up the tree
set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
