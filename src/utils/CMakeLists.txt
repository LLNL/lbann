add_sources(
  quantizer.cpp
  summary.cpp
  random.cpp
  statistics.cpp
  im2col.cpp
  cudnn_wrapper.cpp
  options.cpp
  omp_diagnostics.cpp
)

if(LBANN_HAS_CUDA)
  add_sources(
    cudnn_wrapper.cu
    )
endif()
