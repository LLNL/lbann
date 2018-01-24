# Add the headers for this directory
set_full_path(THIS_DIR_HEADERS
  io_layer.hpp
  )

# Add the subdirectories
add_subdirectory(input)
add_subdirectory(target)

# Propagate the files up the tree
set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
