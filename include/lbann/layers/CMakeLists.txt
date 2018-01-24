# Add the headers for this directory
set_full_path(THIS_DIR_HEADERS
  layer.hpp
  )

# Add the subdirectories
add_subdirectory(activations)
add_subdirectory(io)
add_subdirectory(learning)
add_subdirectory(regularizers)
add_subdirectory(transform)

# Propagate the files up the tree
set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
