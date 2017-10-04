if(LBANN_HAS_CUDA AND WITH_SOFTMAX_CUDA)  
  add_sources(
    softmax_cuda.cu
  )
endif()
