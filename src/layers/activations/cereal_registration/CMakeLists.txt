# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  elu.cpp
  identity.cpp
  leaky_relu.cpp
  log_softmax.cpp
  relu.cpp
  softmax.cpp
  )

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
