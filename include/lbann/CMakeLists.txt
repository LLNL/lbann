# Add the headers for this directory
set_full_path(THIS_DIR_HEADERS
  Elemental_extensions.hpp
  base.hpp
  comm.hpp
  lbann.hpp
  )

# Add the subdirectories
add_subdirectory(callbacks)
add_subdirectory(data_readers)
add_subdirectory(data_store)
add_subdirectory(io)
add_subdirectory(layers)
add_subdirectory(metrics)
add_subdirectory(models)
add_subdirectory(objective_functions)
add_subdirectory(optimizers)
add_subdirectory(proto)
add_subdirectory(transforms)
add_subdirectory(utils)
add_subdirectory(weights)

# Propagate the files up the tree
set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
