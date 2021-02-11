# Add the headers for this directory
set_full_path(THIS_DIR_HEADERS
  adagrad.hpp
  adagrad_impl.hpp
  adam.hpp
  adam_impl.hpp
  data_type_optimizer.hpp
  data_type_optimizer_impl.hpp
  hypergradient_adam.hpp
  hypergradient_adam_impl.hpp
  optimizer.hpp
  rmsprop.hpp
  rmsprop_impl.hpp
  sgd.hpp
  sgd_impl.hpp
  )

# Propagate the files up the tree
set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
