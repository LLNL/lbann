# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  input_layer.cpp
  )

add_subdirectory(cereal_registration)

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
