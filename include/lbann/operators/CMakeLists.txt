# Add the headers for this directory
set_full_path(THIS_DIR_HEADERS
  elementwise_operator.hpp
  operator.hpp
  )

add_subdirectory(math)

# Propagate the files up the tree
set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
