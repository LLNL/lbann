# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  initializer.cpp
  variance_scaling_initializers.cpp
  weights.cpp
  )

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
