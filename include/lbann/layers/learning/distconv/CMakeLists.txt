set_full_path(THIS_DIR_HEADERS
  distconv_layers.hpp
  )

# Propagate the files up the tree
set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)