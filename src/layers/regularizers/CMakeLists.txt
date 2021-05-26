# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  batch_normalization.cpp
  dropout.cpp
  entrywise_batch_normalization.cpp
  instance_norm.cpp
  layer_norm.cpp
  local_response_normalization.cpp
  selu_dropout.cpp
  )

if (LBANN_HAS_GPU)
  # Add the CUDA source files for this directory
  set_full_path(THIS_DIR_CU_SOURCES
    batch_normalization.cu
    entrywise_batch_normalization.cu
    instance_norm.cu
    layer_norm.cu
    )
endif ()

add_subdirectory(cereal_registration)

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
set(GPU_SOURCES "${GPU_SOURCES}" "${THIS_DIR_CU_SOURCES}" PARENT_SCOPE)
