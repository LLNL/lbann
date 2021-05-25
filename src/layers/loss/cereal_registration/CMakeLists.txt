# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  binary_cross_entropy.cpp
  boolean_accuracy.cpp
  boolean_false_negative.cpp
  boolean_false_positive.cpp
  categorical_accuracy.cpp
  cross_entropy.cpp
  l1_norm.cpp
  l2_norm2.cpp
  mean_absolute_error.cpp
  mean_squared_error.cpp
  sigmoid_binary_cross_entropy.cpp
  top_k_categorical_accuracy.cpp
  )

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
