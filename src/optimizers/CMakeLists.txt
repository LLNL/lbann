add_sources(
  optimizer.cpp
  optimizer_sgd.cpp
  optimizer_adagrad.cpp
  optimizer_rmsprop.cpp
  optimizer_adam.cpp
  optimizer_hypergradient_adam.cpp
)

if(LBANN_HAS_CUDA)
  add_sources(
    optimizer_adam_cuda.cu
  )
endif()
