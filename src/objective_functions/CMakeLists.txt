add_subdirectory(loss_functions)
add_subdirectory(weight_regularization)

add_sources(
  objective_function.cpp
  objective_function_term.cpp
  kl_divergence.cpp
)
