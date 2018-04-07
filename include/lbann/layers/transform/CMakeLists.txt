# Add the headers for this directory
set_full_path(THIS_DIR_HEADERS
  concatenation.hpp
  pooling.hpp
  reshape.hpp
  safe_inv.hpp
  slice.hpp
  split.hpp
  sum.hpp
  transform.hpp
  unpooling.hpp
  constant.hpp
  dummy.hpp
  hadamard.hpp
  gaussian.hpp
  bernoulli.hpp
  uniform.hpp
  )

# Propagate the files up the tree
set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
