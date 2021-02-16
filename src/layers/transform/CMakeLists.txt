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
  gather.cpp
  gather_builder.cpp
  gaussian.cpp
  hadamard.cpp
  in_top_k.cpp
  pooling.cpp
  reduction.cpp
  reshape.cpp
  scatter.cpp
  scatter_builder.cpp
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

if (LBANN_HAS_GPU)
  # Add the CUDA source files for this directory
  set_full_path(THIS_DIR_CU_SOURCES
    concatenate.cu
    crop.cu
    gather.cu
    in_top_k.cu
    sort.cu
    scatter.cu
    slice.cu
    tessellate.cu
    split.cu
    sum.cu
    )
endif ()

# Add the subdirectories
add_subdirectory(cereal_registration)

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
set(GPU_SOURCES "${GPU_SOURCES}" "${THIS_DIR_CU_SOURCES}" PARENT_SCOPE)
