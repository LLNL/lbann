# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  binary_cross_entropy.cpp
  boolean_accuracy.cpp
  boolean_false_negative.cpp
  boolean_false_positive.cpp
  sigmoid_binary_cross_entropy.cpp
  )

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
