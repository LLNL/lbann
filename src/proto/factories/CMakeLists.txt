# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  callback_factory.cpp
  layer_factory.cpp
  layer_graph_factory.cpp
  model_factory.cpp
  objective_function_factory.cpp
  optimizer_factory.cpp
  trainer_factory.cpp
  weights_factory.cpp
)

if (LBANN_HAS_OPENCV)
  list(APPEND THIS_DIR_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/transform_factory.cpp")
endif ()


# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
