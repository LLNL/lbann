# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  fan_in_fan_out_initializers.cpp
  initializer.cpp
  weights.cpp
  )

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
