# Add the headers for this directory
set_full_path(THIS_DIR_HEADERS
  binary_cross_entropy.hpp
  cross_entropy.hpp
  cross_entropy_with_uncertainty.hpp
  geom_negloglike.hpp
  loss_function.hpp
  mean_absolute_deviation.hpp
  mean_squared_error.hpp
  poisson_negloglike.hpp
  polya_negloglike.hpp
  )

# Propagate the files up the tree
set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
