# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  bilinear_resize.cpp
  rotation.cpp
  )

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)

