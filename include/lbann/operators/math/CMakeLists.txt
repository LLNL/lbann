# Add the headers for this directory
set_full_path(THIS_DIR_HEADERS
  clamp.hpp
  math_builders.hpp
  math_builders_impl.hpp
  )

# Propagate the files up the tree
set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
