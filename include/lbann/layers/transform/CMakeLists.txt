# Add the headers for this directory
set_full_path(THIS_DIR_HEADERS
  concatenation.hpp
  noise.hpp
  pooling.hpp
  reshape.hpp
  slice.hpp
  split.hpp
  sum.hpp
  transform.hpp
  unpooling.hpp
  )

# Propagate the files up the tree
set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
