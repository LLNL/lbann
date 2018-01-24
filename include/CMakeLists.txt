# Add the subdirectories
add_subdirectory(lbann)

# Propagate the files up the tree
set(LBANN_HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
