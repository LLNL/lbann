# Add the headers for this directory
set_full_path(THIS_DIR_HEADERS
  metric.hpp
  metric_categorical_accuracy.hpp
  metric_mean_absolute_deviation.hpp
  metric_mean_squared_error.hpp
  metric_pearson_correlation.hpp
  metric_top_k_categorical_accuracy.hpp
  )

# Propagate the files up the tree
set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
