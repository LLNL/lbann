# Add the headers for this directory
set_full_path(THIS_DIR_HEADERS
  cross_entropy.hpp
  l2_norm2.hpp
  mean_squared_error.hpp
  top_k_categorical_accuracy.hpp
  )

# Propagate the files up the tree
set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
