# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  group_lasso.cpp
  l1.cpp
  l2.cpp
  )

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
