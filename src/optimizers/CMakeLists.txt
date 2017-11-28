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
    adam.cu
    sgd.cu
  )
endif()
