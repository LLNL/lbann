add_sources(
  optimizer.cpp
  sgd.cpp
  adagrad.cpp
  rmsprop.cpp
  adam.cpp
  hypergradient_adam.cpp
)

if(LBANN_HAS_CUDA)
  add_sources(
#    optimizer_adam_cuda.cu
  )
endif()
