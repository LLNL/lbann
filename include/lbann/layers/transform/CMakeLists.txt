# Add the headers for this directory
set_full_path(THIS_DIR_HEADERS
  concatenation.hpp
  pooling.hpp
  reshape.hpp
  slice.hpp
  split.hpp
  sum.hpp
  weighted_sum.hpp
  transform.hpp
  unpooling.hpp
  constant.hpp
  dummy.hpp
  hadamard.hpp
  reduction.hpp
  evaluation.hpp
  gaussian.hpp
  bernoulli.hpp
  uniform.hpp
  crop.hpp
  categorical_random.hpp
  discrete_random.hpp
  stop_gradient.hpp
  in_top_k.hpp
  sort.hpp
  weights.hpp
  )

# Propagate the files up the tree
set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
