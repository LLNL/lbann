if (LBANN_HAS_CUDA AND LBANN_SOFTMAX_CUDA)
  # Add the CUDA source files for this directory
  set_full_path(THIS_DIR_CU_SOURCES
    softmax_cuda.cu
    )
endif ()

# Propagate the files up the tree
set(CUDA_SOURCES "${CUDA_SOURCES}" "${THIS_DIR_CU_SOURCES}" PARENT_SCOPE)
