# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  callback_factory.cpp
  layer_factory.cpp
  layer_graph_factory.cpp
  model_factory.cpp
  objective_function_factory.cpp
  optimizer_factory.cpp
  transform_factory.cpp
  weights_factory.cpp
)

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
