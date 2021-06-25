# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  factory.cpp
  ltfb.cpp
  sgd_training_algorithm.cpp
  training_algorithm.cpp
  )

# Add the LTFB support class implementations
add_subdirectory(ltfb)

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
