# Add the headers for this directory
set_full_path(THIS_DIR_HEADERS
  activation.hpp
  atan.hpp
  bent_identity.hpp
  elu.hpp
  exponential.hpp
  identity.hpp
  leaky_relu.hpp
  relu.hpp
  selu.hpp
  sigmoid.hpp
  smooth_relu.hpp
  softmax.hpp
  softplus.hpp
  swish.hpp
  tanh.hpp
  power.hpp
  )

# Propagate the files up the tree
set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
