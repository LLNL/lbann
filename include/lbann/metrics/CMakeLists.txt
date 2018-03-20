# Add the headers for this directory
set_full_path(THIS_DIR_HEADERS
  metric.hpp
  categorical_accuracy.hpp
  mean_absolute_deviation.hpp
  mean_squared_error.hpp
  pearson_correlation.hpp
  top_k_categorical_accuracy.hpp
  r2.hpp
  boolean_accuracy.hpp
  boolean_false_positives.hpp
  boolean_false_negatives.hpp
  )

# Propagate the files up the tree
set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
