# Add the headers for this directory
set_full_path(THIS_DIR_HEADERS
  activations.hpp
  elu.hpp
  identity.hpp
  leaky_relu.hpp
  log_softmax.hpp
  softmax.hpp
  )

if (LBANN_HAS_DISTCONV)
  list(APPEND THIS_DIR_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/relu.hpp")
endif ()

# Propagate the files up the tree
set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
