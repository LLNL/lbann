# Add the headers for this directory
set_full_path(THIS_DIR_HEADERS
  data_type_weights.hpp
  initializer.hpp
  variance_scaling_initializers.hpp
  weights.hpp
  weights_helper.hpp
  )

# Propagate the files up the tree
set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
