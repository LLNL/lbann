# Add the headers for this directory
set_full_path(THIS_DIR_HEADERS
  bilinear_resize.hpp
  rotation.hpp
  )

# Propagate the files up the tree
set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
