# Add the headers for this directory
set_full_path(THIS_DIR_HEADERS
  batch_functional_inference_algorithm.hpp
  kfac.hpp
  ltfb.hpp
  sgd_training_algorithm.hpp
  training_algorithm.hpp
  )

add_subdirectory(kfac)

# Propagate the files up the tree
set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
