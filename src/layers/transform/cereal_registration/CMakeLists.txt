# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  bernoulli.cpp
  categorical_random.cpp
  concatenate.cpp
  concatenate_builder.cpp
  constant.cpp
  crop.cpp
  crop_builder.cpp
  discrete_random.cpp
  dummy.cpp
  evaluation.cpp
  gaussian.cpp
  hadamard.cpp
  in_top_k.cpp
  pooling.cpp
  reduction.cpp
  reshape.cpp
  slice.cpp
  sort.cpp
  split.cpp
  stop_gradient.cpp
  sum.cpp
  tessellate.cpp
  uniform.cpp
  unpooling.cpp
  weighted_sum.cpp
  weights.cpp
  )

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)

