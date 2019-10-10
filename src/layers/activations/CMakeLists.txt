# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  activations.cpp
  elu.cpp
  identity.cpp
  leaky_relu.cpp
  log_softmax.cpp
  softmax.cpp
  )

if (LBANN_HAS_CUDA)
  # Add the CUDA source files for this directory
  set_full_path(THIS_DIR_CU_SOURCES
    activations.cu
    elu.cu
    leaky_relu.cu
    log_softmax.cu
    softmax.cu
    )
endif ()

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
set(CUDA_SOURCES "${CUDA_SOURCES}" "${THIS_DIR_CU_SOURCES}" PARENT_SCOPE)
