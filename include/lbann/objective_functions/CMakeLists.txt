# Add the headers for this directory
set_full_path(THIS_DIR_HEADERS
  objective_function.hpp
  objective_function_term.hpp
  )

# Add the subdirectories
add_subdirectory(loss_functions)
add_subdirectory(weight_regularization)

# Propagate the files up the tree
set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
