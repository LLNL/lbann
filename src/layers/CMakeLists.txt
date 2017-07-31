add_sources(
  layer.cpp
)

if(LBANN_HAS_CUDA AND WITH_FULLY_CONNECTED_CUDA)
  add_sources(
    fully_connected_cuda.cu
  )
endif()
