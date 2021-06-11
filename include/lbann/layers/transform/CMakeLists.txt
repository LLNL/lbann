# Add the headers for this directory
set_full_path(THIS_DIR_HEADERS
  batchwise_reduce_sum.hpp
  concatenate.hpp
  cross_grid_sum_slice.hpp
  cross_grid_sum.hpp
  pooling.hpp
  reshape.hpp
  slice.hpp
  split.hpp
  sum.hpp
  weighted_sum.hpp
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
  tessellate.hpp
  scatter.hpp
  gather.hpp
  )

# Propagate the files up the tree
set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
