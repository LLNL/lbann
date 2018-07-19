# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  binary_cross_entropy.cpp
  cross_entropy.cpp
  cross_entropy_with_uncertainty.cpp
  geom_negloglike.cpp
  loss_function.cpp
  mean_absolute_deviation.cpp
  mean_squared_error.cpp
  mean_absolute_error.cpp
  poisson_negloglike.cpp
  polya_negloglike.cpp
  )

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
