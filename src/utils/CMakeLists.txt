# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  cudnn_wrapper.cpp
  file_utils.cpp
  im2col.cpp
  omp_diagnostics.cpp
  options.cpp
  protobuf_utils.cpp
  quantizer.cpp
  random.cpp
  stack_trace.cpp
  statistics.cpp
  summary.cpp
)

if (LBANN_HAS_CUDA)
  # Add the CUDA source files for this directory
  set_full_path(THIS_DIR_CU_SOURCES
    cudnn_wrapper.cu
    )
endif ()

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
set(CUDA_SOURCES "${CUDA_SOURCES}" "${THIS_DIR_CU_SOURCES}" PARENT_SCOPE)
