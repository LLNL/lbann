# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  relu.cpp
  sigmoid.cpp
  softmax.cpp
  logsoftmax.cpp
  )

if (LBANN_HAS_CUDA)
  # Add the CUDA source files for this directory
  set_full_path(THIS_DIR_CU_SOURCES
    relu.cu
    sigmoid.cu
    softmax.cu
    logsoftmax.cu
    )
endif ()

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
set(CUDA_SOURCES "${CUDA_SOURCES}" "${THIS_DIR_CU_SOURCES}" PARENT_SCOPE)
