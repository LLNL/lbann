# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  bernoulli.cpp
  categorical_random.cpp
  concatenation.cpp
  constant.cpp
  crop.cpp
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

if (LBANN_HAS_CUDA)
  # Add the CUDA source files for this directory
  set_full_path(THIS_DIR_CU_SOURCES
    crop.cu
    in_top_k.cu
    sort.cu
    tessellate.cu
    )
endif ()

if (LBANN_HAS_DISTCONV)
  list(APPEND THIS_DIR_CU_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/split.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/sum.cu")
endif ()

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
set(CUDA_SOURCES "${CUDA_SOURCES}" "${THIS_DIR_CU_SOURCES}" PARENT_SCOPE)
