# Add the headers for this directory
set_full_path(THIS_DIR_HEADERS
  batch_normalization.hpp
  dropout.hpp
  local_response_normalization.hpp
  regularizer.hpp
  selu_dropout.hpp
  )

# Propagate the files up the tree
set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
