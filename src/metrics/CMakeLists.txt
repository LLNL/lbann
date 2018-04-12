# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  categorical_accuracy.cpp
  mean_absolute_deviation.cpp
  mean_squared_error.cpp
  metric.cpp
  pearson_correlation.cpp
  top_k_categorical_accuracy.cpp
  r2.cpp
  boolean_accuracy.cpp
  boolean_false_positives.cpp
  boolean_false_negatives.cpp
  )

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
