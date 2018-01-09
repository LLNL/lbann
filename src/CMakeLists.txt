add_subdirectory(data_distributions)
add_subdirectory(data_readers)
add_subdirectory(data_store)
add_subdirectory(layers)
add_subdirectory(models)
add_subdirectory(weights)
add_subdirectory(optimizers)
add_subdirectory(callbacks)
add_subdirectory(io)
add_subdirectory(utils)
add_subdirectory(objective_functions)
add_subdirectory(proto)
add_subdirectory(metrics)

add_sources(
  Elemental_extensions.cpp 
  base.cpp
  params.cpp
  comm.cpp
)

