# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  data_type_layer.cpp
  layer.cpp
  operator_layer.cpp
  )

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
